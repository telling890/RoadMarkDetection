param(
    [int]$DebounceSeconds = 30
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$syncScript = Join-Path $PSScriptRoot "sync_github.ps1"
$lockFile = Join-Path $root ".github-sync.lock"
$excludedPattern = "[\\/](\.git|\.idea|\.github-sync-worktree|\.upload-worktree|__pycache__|dataset|runs|annotations|new data1)[\\/]"
$watcher = [IO.FileSystemWatcher]::new($root)
$watcher.IncludeSubdirectories = $true
$watcher.NotifyFilter = [IO.NotifyFilters]"FileName, DirectoryName, LastWrite"
$watcher.EnableRaisingEvents = $true
$pending = $false
$lastChange = Get-Date

Write-Output "GitHub sync watcher started: $root"
try {
    while ($true) {
        $change = $watcher.WaitForChanged([IO.WatcherChangeTypes]::All, 2000)
        if (-not $change.TimedOut) {
            $fullPath = Join-Path $root $change.Name
            if ($fullPath -ne $lockFile -and $fullPath -notmatch $excludedPattern -and $fullPath -notlike "*.pyc" -and $fullPath -notlike "*.cache") {
                $pending = $true
                $lastChange = Get-Date
            }
        }
        if ($pending -and ((Get-Date) - $lastChange).TotalSeconds -ge $DebounceSeconds) {
            try {
                & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $syncScript
            }
            catch {
                Write-Error $_
            }
            $pending = $false
        }
    }
}
finally {
    $watcher.Dispose()
}
