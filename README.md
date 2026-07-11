# 路面标线缺失检测系统

本项目用于开展“复杂道路环境下路面标线缺失检测”实验。任务目标是在道路图像、视频或摄像头画面中检测标线缺失、断裂、严重磨损、遮挡导致不可见等异常区域，为道路养护巡检和交通基础设施状态评估提供自动化识别结果。

项目采用 YOLO26 作为检测基线，并围绕缺失标线的细长形态、小目标、多尺度、低对比度和不规则边界问题，引入 EMA 注意力、BiFPN 多尺度融合、C2f-DCN 形变建模和 Wise-IoU 定位损失进行改进。

## 1. 任务定义

本仓库主线是目标检测任务，不是分类任务。模型输入为道路图像，输出为缺失或退化标线区域的边界框、类别和置信度。

目标数据协议暂定为以下 10 类路面标线缺失目标：

```text
0 lane_line_missing
1 lane_line_break
2 edge_line_missing
3 stop_line_missing
4 crosswalk_missing
5 arrow_missing
6 guide_line_missing
7 worn_marking_missing
8 occluded_marking_missing
9 other_marking_missing
```

### 数据语义警告

当前 `dataset/` 的标签来自道路病害数据集 `D00-D90`。`data/road_mark.yaml` 目前只改变了类别显示名，没有把道路裂缝、坑槽等标注转换成真实的路面标线缺失标注。改名不能改变标签语义。

因此，现有数据只能用于检查训练、验证、模型模块和 GPU 链路，不能把其指标作为“路面标线缺失识别准确率”写入论文。正式实验前必须逐图复核或重新标注目标框，并将 [data/label_semantics.yaml](data/label_semantics.yaml) 中的 `verified_for_target_task` 改为 `true`。训练脚本会在该字段为 `false` 时打印警告。

## 2. 项目结构

```text
RoadMarkDetection/
├── data/
│   ├── road_mark.yaml
│   └── */*_txt.zip
├── dataset/
│   ├── images/train
│   ├── images/val
│   ├── labels/train
│   └── labels/val
├── losses/
├── models/
│   ├── modules/
│   ├── register_ultralytics.py
│   ├── yolo26_roadmark_ema.yaml
│   ├── yolo26_roadmark_ema_bifpn.yaml
│   └── yolo26_roadmark_full.yaml
├── roadmark_experiments/
├── utils/
├── train.py
├── val.py
├── detect.py
├── app.py
├── experiment.py
├── prepare_dataset.py
├── README.md
└── readme-experiment.md
```

## 3. 环境安装

建议使用 Python 3.10 和 CUDA 版 PyTorch。

```bash
pip install -r requirements.txt
```

检查 GPU：

```bash
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

当前已验证环境：

```text
torch: 2.11.0+cu128
CUDA: 12.8
GPU: NVIDIA GeForce RTX 5070 Laptop GPU
```

## 4. 数据准备

### 4.1 当前数据复核结论

当前 `dataset/` 在文件格式上可训练，但其 `D00-D90` 框主要是裂缝、龟裂、坑槽和修补区域，不是路面标线缺失真值。该目录只能用于代码链路测试，不能用于正式路面标线缺失实验。

### 4.2 人工标注新的单类数据集

正式任务采用单类 `road_mark_missing`。标注框只覆盖能够从上下文确认的缺失、断裂或严重磨损标线区段；不要框裂缝、坑槽、阴影、车辆遮挡、完整标线或本来没有标线的路面。

首批 800 张候选清单位于 `annotations/road_mark_missing/manifest.csv`。重新筛选候选：

```bash
python annotate.py select --source dataset/images --workspace annotations/road_mark_missing --max-images 800 --scan-limit 6000
```

#### 使用 LabelImg 打开标注程序

在 Windows PowerShell 中进入项目并激活环境：

```powershell
conda activate py320
cd C:\Users\lenovo\Desktop\RoadMarkDetection
python annotate.py review --workspace annotations\road_mark_missing
```

如果 `python` 没有使用 `py320` 环境，直接指定解释器：

```powershell
D:\conda\envs\py320\python.exe annotate.py review --workspace annotations\road_mark_missing
```

该命令会自动准备候选图片、启动 LabelImg，并设置为 YOLO、单类别和自动保存模式。类别固定为 `road_mark_missing`。

LabelImg 常用操作：

| 操作 | 按键或鼠标 |
|---|---|
| 新建标注框 | `W`，然后鼠标左键拖动 |
| 保存当前标签 | `Ctrl+S` |
| 删除选中的框 | `Delete` |
| 上一张 / 下一张 | `A` / `D` |
| 放大 / 缩小 | `Ctrl++` / `Ctrl+-` |
| 关闭程序 | 关闭 LabelImg 窗口 |

关闭 LabelImg 后，程序会自动把新增或修改过的 YOLO `.txt` 同步回标注清单。没有框的图片不会自动判为负样本，以免把尚未查看的图片误标为负样本。

确认已经逐张查看全部剩余图片后，才能执行：

```powershell
python annotate.py labelimg-sync --workspace annotations\road_mark_missing --accept-unlabeled-negative
```

该命令会把仍然没有标签文件的图片确认为负样本。若还没有检查完全部图片，不要使用 `--accept-unlabeled-negative`。

查看标注进度：

```powershell
python annotate.py status --workspace annotations\road_mark_missing
```

标注记录保存在 `annotations/road_mark_missing/`。LabelImg 无法启动时，可使用备用工具：

```powershell
python annotate.py review-native --workspace annotations\road_mark_missing
```

备用工具中使用鼠标左键框选，`S` 保存正样本，`N` 确认负样本，`Z` 撤销，`A/D` 切换图片，`Q` 退出。

完成复核后导出到用户指定的 `new data1/`：

```bash
python annotate.py export --workspace annotations/road_mark_missing --output "new data1" --data data/road_mark_missing.yaml --train-ratio 0.8 --force
```

导出结果：

```text
new data1/
├── images/train
├── images/val
├── labels/train
└── labels/val
```

建议正式训练前至少完成 300 张正样本和 100 张负样本；论文实验建议达到 1000 张以上正样本，并由第二人抽检不少于 10%。

后续自采图片可持续追加，不会覆盖已有标注：

```bash
python annotate.py ingest --source incoming_data/session01 --batch session01 --scene urban --weather sunny --time-of-day day
python annotate.py status
```

训练出第一版单类模型后，可用它生成预标注，再逐图确认：

```bash
python annotate.py prelabel --weights runs/train/exp_baseline_ciou/weights/best.pt --device 0
python annotate.py review
```

完整的拍摄配额、目录命名、质量门槛和增量流程见 [DATA_COLLECTION.md](DATA_COLLECTION.md)。导出会按采集会话分组划分 train/val，避免相邻视频帧跨集合导致指标虚高。

新数据审计和训练：

```bash
python experiment.py --exp EXP-00 --data data/road_mark_missing.yaml --outputs runs/roadmark_missing_audit
python experiment.py --exp EXP-01 --data data/road_mark_missing.yaml --device 0 --profile accuracy
```

### 4.3 原始道路病害数据重新导入

原始数据放在 `data/` 目录。工程支持 Roboflow YOLO txt zip 数据包，能自动处理 `labels` 和 `labelTxt` 两种标签目录。

重新生成训练集和验证集：

```bash
python prepare_dataset.py --raw-data data --prepared-data dataset --data data/road_mark.yaml --outputs runs/experiments --force-selection --import-zips
```

如果 `data/` 中原始 zip 已移动，但 `runs/_extracted_txt` 仍保留了解压数据，可用缓存恢复：

```bash
python prepare_dataset.py --raw-data runs/_extracted_txt --prepared-data dataset --data data/road_mark.yaml --outputs runs/experiments --force-selection --import-zips
```

生成后结构为：

```text
dataset/
├── images/train/
├── images/val/
├── labels/train/
└── labels/val/
```

YOLO 标签格式：

```text
class x_center y_center width height
```

数据配置文件：

```text
data/road_mark.yaml
```

当前已导入数据统计：

```text
train: 19302 images, 64054 boxes
val:   4127 images, 13953 boxes
missing labels: 0
invalid labels: 0
```

## 5. 完整实验执行流程（按顺序）

下面的步骤应从 Step 1 依次执行。正式实验统一使用人工复核后的 `data/road_mark_missing.yaml`，不要使用道路病害来源的 `data/road_mark.yaml` 生成论文指标。

### Step 1：安装环境并确认 GPU

目的：确保 Python、PyTorch、CUDA 和依赖可用。

```powershell
conda activate py320
cd C:\Users\lenovo\Desktop\RoadMarkDetection
pip install -r requirements.txt
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

完成标准：`torch.cuda.is_available()` 输出 `True`，并显示 NVIDIA GPU 名称。

### Step 2：导入后续自采图片

目的：对新图片去重、检查分辨率，并记录场景、天气和采集时段。首次只使用现有 800 张候选时可以跳过本步骤。

```powershell
python annotate.py ingest --source incoming_data\session01 --batch session01 --scene urban --weather sunny --time-of-day day
python annotate.py status
```

产物：

- `annotations/road_mark_missing/manifest.csv`
- `annotations/road_mark_missing/source_images/`
- `annotations/road_mark_missing/ingest_rejected.csv`

完成标准：新增图片进入 `pending`，重复图被跳过，无法读取或分辨率不足的图片进入拒绝清单。

### Step 3：人工标注正样本和负样本

目的：生成真实的单类 `road_mark_missing` 边界框。

```powershell
python annotate.py review --workspace annotations\road_mark_missing
```

在 LabelImg 中按 `W` 后拖动框选，按 `Ctrl+S` 保存，使用 `A/D` 切换图片。关闭 LabelImg 后会自动同步已保存标签。确认所有剩余无框图片均已人工查看后，再将其标记为负样本：

```powershell
python annotate.py labelimg-sync --workspace annotations\road_mark_missing --accept-unlabeled-negative
```

查看进度：

```powershell
python annotate.py status --workspace annotations\road_mark_missing
```

完成标准：最低应有 300 张 `positive` 和 100 张 `negative`；论文主实验建议至少 1000 张正样本。所有待导出图片必须经过人工确认，不能保持 `pending` 或 `prelabel`。

### Step 4：导出新数据集到 new data1

目的：只导出人工确认的正负样本，并按采集组划分 train/val，避免相邻帧泄漏。

```powershell
python annotate.py export --workspace annotations\road_mark_missing --output "new data1" --data data/road_mark_missing.yaml --train-ratio 0.8 --seed 42 --force
```

产物：

```text
new data1/
├── images/train
├── images/val
├── labels/train
└── labels/val
```

完成标准：train 和 val 都非空，正样本含 YOLO 框，负样本对应空标签文件。

### Step 5：执行 EXP-00 数据审计

目的：检查图片和标签是否一一对应、坐标是否合法、类别编号是否为 0。

```powershell
python experiment.py --exp EXP-00 --data data/road_mark_missing.yaml --outputs runs/roadmark_missing_audit
```

主要产物：

- `runs/roadmark_missing_audit/dataset_audit.md`
- `runs/roadmark_missing_audit/dataset_stats.csv`
- `runs/roadmark_missing_audit/class_distribution.csv`
- `runs/roadmark_missing_audit/invalid_labels.csv`
- `runs/roadmark_missing_audit/missing_labels.json`

完成标准：`missing labels=0`、`invalid lines=0`，并且语义状态为人工标注已验证。

### Step 6：执行代码和 GPU 完整性检查

目的：在正式训练前检查数据工具、EMA、BiFPN、C2f-DCN、Wise-IoU 和 CUDA 前向链路。

```powershell
python tests/smoke_test.py
python tests/integrity_test.py --require-gpu --data data/road_mark_missing.yaml
python experiment.py --exp all --data data/road_mark_missing.yaml --device 0 --profile accuracy --dry-run
```

完成标准：两个测试均显示 `passed`，dry-run 只打印命令且不创建实验结果。

### Step 7：执行 1 轮 GPU 冒烟训练

目的：验证新数据能完成 DataLoader、前向传播、反向传播和权重保存。该结果不能写入论文。

```powershell
python train.py --variant baseline --loss ciou --profile standard --data data/road_mark_missing.yaml --epochs 1 --batch 4 --img-size 320 --device 0 --workers 0 --project runs/smoke --name roadmark_missing_smoke
```

完成标准：生成 `runs/smoke/roadmark_missing_smoke/weights/best.pt`，且没有 CUDA、标签或类别数异常。

### Step 8：执行 EXP-01 YOLO26 Baseline

目的：建立所有改进模型的基础对照。

```powershell
python train.py --variant baseline --loss ciou --profile accuracy --data data/road_mark_missing.yaml --epochs 200 --device 0 --workers 8 --seed 42 --project runs/train --name exp_baseline_ciou
python val.py --weights runs/train/exp_baseline_ciou/weights/best.pt --data data/road_mark_missing.yaml --img-size 768 --batch 8 --device 0 --project runs/val --name exp_baseline_ciou
```

产物：baseline 最佳权重、`metrics.json`、PR 曲线、混淆矩阵和验证图。

完成标准：训练正常收敛，验证指标不是全零，并记录 Precision、Recall、F1、mAP50、mAP50-95 和 FPS。

### Step 9：执行 EXP-03 EMA 注意力实验

目的：验证 EMA 对低对比度、磨损和弱纹理缺失区域的作用。

```powershell
python train.py --variant ema --loss ciou --profile accuracy --data data/road_mark_missing.yaml --epochs 200 --device 0 --workers 8 --seed 42 --project runs/train --name exp_ema_ciou
python val.py --weights runs/train/exp_ema_ciou/weights/best.pt --data data/road_mark_missing.yaml --img-size 768 --batch 8 --device 0 --project runs/val --name exp_ema_ciou
```

对比对象：`exp_baseline_ciou` 与 `exp_ema_ciou`。至少比较 Recall、mAP50-95 和低对比度样例漏检情况。

### Step 10：执行 EXP-04 EMA + BiFPN 实验

目的：验证多尺度融合对远距离、小尺度缺失标线的作用。

```powershell
python train.py --variant ema_bifpn --loss ciou --profile accuracy --data data/road_mark_missing.yaml --epochs 200 --device 0 --workers 8 --seed 42 --project runs/train --name exp_ema_bifpn_ciou
python val.py --weights runs/train/exp_ema_bifpn_ciou/weights/best.pt --data data/road_mark_missing.yaml --img-size 768 --batch 8 --device 0 --project runs/val --name exp_ema_bifpn_ciou
```

对比对象：`exp_ema_ciou` 与 `exp_ema_bifpn_ciou`。重点检查小框和远距离目标的 Recall。

### Step 11：执行 EXP-05 C2f-DCN 完整结构实验

目的：验证 DCN 对弯道、透视变化和不规则缺失边界的建模收益，此时仍使用 CIoU。

```powershell
python train.py --variant full --loss ciou --profile accuracy --data data/road_mark_missing.yaml --epochs 200 --device 0 --workers 8 --seed 42 --project runs/train --name exp_full_ciou
python val.py --weights runs/train/exp_full_ciou/weights/best.pt --data data/road_mark_missing.yaml --img-size 768 --batch 8 --device 0 --project runs/val --name exp_full_ciou
```

对比对象：`exp_ema_bifpn_ciou` 与 `exp_full_ciou`。同时记录参数量、FPS 和 mAP50-95，避免只增加计算量而没有精度收益。

### Step 12：执行 EXP-06 Wise-IoU 实验

目的：保持完整网络结构不变，只替换定位损失，验证边界框定位质量。

```powershell
python train.py --variant full --loss wise_iou --profile accuracy --data data/road_mark_missing.yaml --epochs 200 --device 0 --workers 8 --seed 42 --project runs/train --name exp_full_wise_iou
python val.py --weights runs/train/exp_full_wise_iou/weights/best.pt --data data/road_mark_missing.yaml --img-size 768 --batch 8 --device 0 --project runs/val --name exp_full_wise_iou
```

对比对象：`exp_full_ciou` 与 `exp_full_wise_iou`。Wise-IoU 的主要判据是 mAP50-95 和框偏移案例是否改善。

### Step 13：执行 EXP-07 消融结果汇总

目的：读取 Step 8-12 已有权重和指标，生成统一消融表和对比图，不重复训练。

```powershell
python experiment.py --exp EXP-07 --data data/road_mark_missing.yaml --device 0 --profile accuracy --skip-train --skip-val --outputs runs/experiments
```

产物：

- `runs/experiments/experiment_results.csv`
- `runs/experiments/experiment_results.md`
- `runs/experiments/figures/figure4_ablation_metrics.png`

完成标准：表中包含 baseline、EMA、EMA+BiFPN、完整结构+CIoU、完整结构+Wise-IoU 五组结果。

### Step 14：执行 EXP-02 复杂环境评估

目的：在夜间、雨天、阴影等独立场景验证集上评估鲁棒性。先分别准备场景 YAML，例如 `data/night.yaml`、`data/rain.yaml` 和 `data/shadow.yaml`。

```powershell
python experiment.py --exp EXP-02 --data data/road_mark_missing.yaml --device 0 --profile accuracy --skip-train --scenario-data night=data/night.yaml --scenario-data rain=data/rain.yaml --scenario-data shadow=data/shadow.yaml --outputs runs/experiments
```

产物：`runs/experiments/complex_env_results.csv`。

完成标准：逐场景报告 Precision、Recall、mAP50、mAP50-95 和 FPS，不用普通 val 指标代替复杂环境指标。

### Step 15：对最终模型执行 TTA 验证

目的：获得精度优先的最终结果。TTA 会降低 FPS，不能与未开启 TTA 的结果混在同一公平对比表中。

```powershell
python val.py --weights runs/train/exp_full_wise_iou/weights/best.pt --data data/road_mark_missing.yaml --img-size 768 --batch 8 --device 0 --tta --project runs/val --name exp_full_wise_iou_tta
```

完成标准：保留普通验证和 TTA 验证两套 `metrics.json`，论文中明确注明是否使用 TTA。

### Step 16：执行 EXP-08 图片和视频推理

图片推理：

```powershell
python experiment.py --exp EXP-08 --device 0 --profile accuracy --weights runs/train/exp_full_wise_iou/weights/best.pt --source test.jpg --outputs runs/experiments
```

视频推理：

```powershell
python experiment.py --exp EXP-08 --device 0 --profile accuracy --weights runs/train/exp_full_wise_iou/weights/best.pt --source road_video.mp4 --outputs runs/experiments
```

摄像头推理：

```powershell
python experiment.py --exp EXP-08 --device 0 --profile accuracy --weights runs/train/exp_full_wise_iou/weights/best.pt --source 0 --show --outputs runs/experiments
```

产物：`runs/detect/roadmark_missing_detect/` 和 `runs/experiments/realtime_results.csv`。

### Step 17：执行 EXP-09 PyQt5 Demo

生成 Demo 检查清单：

```powershell
python experiment.py --exp EXP-09 --weights runs/train/exp_full_wise_iou/weights/best.pt --outputs runs/experiments
```

启动 GUI：

```powershell
python experiment.py --exp EXP-09 --weights runs/train/exp_full_wise_iou/weights/best.pt --launch-gui
```

完成标准：GUI 能加载最终模型，并能处理图片、视频和摄像头输入。

### Step 18：整理最终实验结果

最终至少保留以下内容：

1. 数据审计表和标注规范。
2. 五组消融实验的 Precision、Recall、F1、mAP50、mAP50-95、参数量和 FPS。
3. 普通环境与夜间、雨天、阴影环境结果。
4. PR 曲线、混淆矩阵、训练曲线和消融对比图。
5. 成功检测、漏检和误检可视化案例。
6. 所有实验使用的随机种子、profile、输入尺寸、batch、GPU 和软件版本。

只有人工标注数据通过审计，并且完整模型相对 baseline 在多个随机种子或重复实验中稳定提升，才能把结果写入论文结论。

## 6. 实验设计

实验以“标线缺失检测效果是否提升”为核心。每个改进模块都要回答一个具体问题：

| 实验 | 目标 |
|---|---|
| EXP-00 | 检查缺失标线数据、标签和类别协议 |
| EXP-01 | 建立 YOLO26 缺失检测 baseline |
| EXP-02 | 评估夜间、阴影、雨天、磨损、遮挡等复杂环境 |
| EXP-03 | 验证 EMA 对低对比度缺失区域的表达能力 |
| EXP-04 | 验证 BiFPN 对远距离、小尺度缺失目标的检测收益 |
| EXP-05 | 验证 C2f-DCN 对不规则缺失边界和弯曲标线的建模能力 |
| EXP-06 | 验证 Wise-IoU 对缺失区域框定位质量的影响 |
| EXP-07 | 汇总 baseline、消融模型和完整模型 |
| EXP-08 | 验证图片、视频、摄像头推理 |
| EXP-09 | 验证 PyQt5 可视化展示 |

查看实验清单：

```bash
python experiment.py --list
```

## 7. 模型版本

| variant | 说明 |
|---|---|
| `baseline` | YOLO26 clean baseline |
| `ema` | YOLO26 + EMA |
| `ema_bifpn` | YOLO26 + EMA + BiFPN |
| `full` | YOLO26 + C2f-DCN + EMA + BiFPN |

定位损失：

| loss | 说明 |
|---|---|
| `ciou` | baseline 定位损失 |
| `wise_iou` | 改进定位损失 |

训练完整模型：

```bash
python train.py --variant full --loss wise_iou --profile accuracy --data data/road_mark_missing.yaml --device 0
```

训练配置：

| profile | 用途 | 默认输入尺寸 | 默认 batch | 主要策略 |
|---|---|---:|---:|---|
| `standard` | 快速复现和链路检查 | 640 | 16 | 常规增强 |
| `accuracy` | 正式精度实验，默认值 | 768 | 8 | cosine LR、multi-scale、轻量 MixUp、弱化 Mosaic |

同一组消融实验必须使用相同 profile、随机种子和数据划分。显存不足时优先减小 `--batch`，不要先降低输入尺寸。

独立验证并导出逐类别指标：

```bash
python val.py --weights runs/train/exp_full_wise_iou/weights/best.pt --data data/road_mark_missing.yaml --img-size 768 --batch 8 --device 0
```

最终模型可额外进行 TTA 验证；对比表中的所有模型必须统一是否启用 TTA：

```bash
python val.py --weights runs/train/exp_full_wise_iou/weights/best.pt --data data/road_mark_missing.yaml --img-size 768 --batch 8 --device 0 --tta
```

## 8. 结果产物

| 文件 | 内容 |
|---|---|
| `runs/train/<name>/weights/best.pt` | 每组训练最佳权重 |
| `runs/val/<name>/metrics.json` | 验证指标 |
| `runs/val/<name>/confusion_matrix.png` | 混淆矩阵 |
| `runs/val/<name>/PR_curve.png` | PR 曲线 |
| `runs/detect/roadmark_missing_detect/` | 图片、视频或摄像头缺失检测结果 |
| `runs/experiments/dataset_stats.csv` | 数据集统计 |
| `runs/experiments/class_distribution.csv` | 类别分布 |
| `runs/experiments/experiment_results.csv` | 消融实验表 |
| `runs/experiments/figures/figure4_ablation_metrics.png` | 消融对比图 |

## 9. 论文图表建议

| 图表 | 内容 |
|---|---|
| Figure 1 | 路面标线缺失检测系统总体流程 |
| Figure 2 | 改进 YOLO26 网络结构 |
| Figure 3 | 复杂环境下标线缺失检测可视化 |
| Figure 4 | 模型消融指标对比 |
| Figure 5 | 实时推理或 GUI 检测效果 |
| Table 1 | 缺失标线数据集统计 |
| Table 2 | 模型消融实验结果 |
| Table 3 | 复杂环境鲁棒性结果 |
| Table 4 | 实时检测性能 |

## 10. Go / No-Go 判据

继续推进完整系统展示：

- 完整模型相对 YOLO26 baseline 在 Recall、mAP50 或 mAP50-95 上有稳定提升。
- 复杂环境下缺失标线漏检减少。
- 缺失区域定位框更贴合真实缺失边界。
- FPS 满足图片、视频或摄像头演示需求。

需要降级为基础检测系统：

- 完整模型只增加参数量和耗时，没有稳定精度收益。
- Wise-IoU 导致定位指标下降。
- 复杂环境增强明显伤害 clean validation。
- 实时推理或 GUI 无法稳定运行。

## 11. 自检

```bash
python tests/smoke_test.py
python tests/integrity_test.py --require-gpu
python experiment.py --exp all --device 0 --profile accuracy --dry-run
```

`smoke_test.py` 检查模块、数据审计和 zip 导入逻辑；`integrity_test.py` 使用 CUDA 对三个自定义模型执行真实前向，并检查类别数、Wise-IoU 注入和精度配置。正式训练前两项都必须通过。
