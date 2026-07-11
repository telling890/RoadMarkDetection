param(
    [string]$CommitMessage = "",
    [string]$PythonExe = "D:\conda\envs\py320\python.exe"
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$worktree = Join-Path $root ".github-sync-worktree"
$lockFile = Join-Path $root ".github-sync.lock"
$managedDirectories = @("losses", "models", "roadmark_experiments", "tests", "tools", "utils")

function Invoke-Git {
    param([string]$WorkingDirectory, [Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    & git -C $WorkingDirectory @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "git $($Arguments -join ' ') failed with exit code $LASTEXITCODE"
    }
}

if (Test-Path -LiteralPath $lockFile) {
    Write-Output "GitHub sync is already running; skipping this run."
    exit 0
}

try {
    New-Item -ItemType File -Path $lockFile -ErrorAction Stop | Out-Null
    if (Test-Path -LiteralPath $worktree) {
        throw "Temporary worktree already exists: $worktree"
    }

    Invoke-Git $root fetch origin
    Invoke-Git $root worktree add --detach $worktree origin/main
    $resolvedWorktree = (Resolve-Path -LiteralPath $worktree).Path
    if (-not $resolvedWorktree.StartsWith($root + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Temporary worktree is outside the project root: $resolvedWorktree"
    }

    foreach ($directory in $managedDirectories) {
        $source = Join-Path $root $directory
        $target = Join-Path $resolvedWorktree $directory
        if (Test-Path -LiteralPath $target) {
            if (-not $target.StartsWith($resolvedWorktree + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
                throw "Refusing to clean a path outside the temporary worktree: $target"
            }
            Remove-Item -LiteralPath $target -Recurse -Force
        }
        if (Test-Path -LiteralPath $source) {
            Copy-Item -LiteralPath $source -Destination $target -Recurse -Force
        }
    }

    $topLevelFiles = Get-ChildItem -LiteralPath $root -File | Where-Object {
        $_.Name -eq ".gitignore" -or
        $_.Name -eq "requirements.txt" -or
        $_.Extension -in @(".py", ".md")
    }
    foreach ($file in $topLevelFiles) {
        Copy-Item -LiteralPath $file.FullName -Destination (Join-Path $resolvedWorktree $file.Name) -Force
    }

    $dataTarget = Join-Path $resolvedWorktree "data"
    New-Item -ItemType Directory -Path $dataTarget -Force | Out-Null
    Get-ChildItem -LiteralPath (Join-Path $root "data") -File -Filter "*.yaml" |
        Copy-Item -Destination $dataTarget -Force

    Invoke-Git $resolvedWorktree add --all
    $staged = @(& git -C $resolvedWorktree diff --cached --name-only)
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to read staged sync files."
    }
    $forbidden = @($staged | Where-Object {
        $_ -match "^(dataset/|runs/|annotations/|new data1/|\.idea/)" -or
        $_ -match "\.(pt|pth|onnx|engine|zip|cache|pyc)$"
    })
    if ($forbidden.Count -gt 0) {
        throw "Forbidden upload paths found: $($forbidden -join ', ')"
    }
    if ($staged.Count -eq 0) {
        Write-Output "GitHub is already up to date."
        exit 0
    }

    if (-not (Test-Path -LiteralPath $PythonExe)) {
        $PythonExe = (Get-Command python -ErrorAction Stop).Source
    }
    & $PythonExe -m compileall -q $resolvedWorktree
    if ($LASTEXITCODE -ne 0) {
        throw "Python compile check failed; push cancelled."
    }
    & $PythonExe (Join-Path $resolvedWorktree "tests\smoke_test.py")
    if ($LASTEXITCODE -ne 0) {
        throw "Smoke test failed; push cancelled."
    }

    if (-not $CommitMessage) {
        $CommitMessage = "Auto-sync " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
    }
    Invoke-Git $resolvedWorktree commit -m $CommitMessage
    Invoke-Git $resolvedWorktree push origin "HEAD:main"
    Write-Output "GitHub sync completed: $CommitMessage"
}
finally {
    if (Test-Path -LiteralPath $worktree) {
        & git -C $root worktree remove $worktree --force
    }
    Remove-Item -LiteralPath $lockFile -Force -ErrorAction SilentlyContinue
}
