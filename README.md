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

## 5. 快速运行

完整性检查：

```bash
python tests/smoke_test.py
python tests/integrity_test.py --require-gpu
```

数据审计：

```bash
python experiment.py --exp EXP-00 --outputs runs/experiments
```

训练缺失检测 baseline：

```bash
python experiment.py --exp EXP-01 --device 0 --profile accuracy
```

只跑 1 轮检查 GPU 训练链路：

```bash
python experiment.py --exp EXP-01 --device 0 --epochs 1 --batch 8 --workers 4
```

运行完整消融实验：

```bash
python experiment.py --exp EXP-07 --device 0 --profile accuracy
```

图片推理：

```bash
python experiment.py --exp EXP-08 --device 0 --weights runs/train/best.pt --source test.jpg
```

启动 Demo：

```bash
python experiment.py --exp EXP-09 --launch-gui
```

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
python train.py --variant full --loss wise_iou --profile accuracy --data data/road_mark.yaml --device 0
```

训练配置：

| profile | 用途 | 默认输入尺寸 | 默认 batch | 主要策略 |
|---|---|---:|---:|---|
| `standard` | 快速复现和链路检查 | 640 | 16 | 常规增强 |
| `accuracy` | 正式精度实验，默认值 | 768 | 8 | cosine LR、multi-scale、轻量 MixUp、弱化 Mosaic |

同一组消融实验必须使用相同 profile、随机种子和数据划分。显存不足时优先减小 `--batch`，不要先降低输入尺寸。

独立验证并导出逐类别指标：

```bash
python val.py --weights runs/train/exp_full_wise_iou/weights/best.pt --data data/road_mark.yaml --img-size 768 --batch 8 --device 0
```

最终模型可额外进行 TTA 验证；对比表中的所有模型必须统一是否启用 TTA：

```bash
python val.py --weights runs/train/exp_full_wise_iou/weights/best.pt --data data/road_mark.yaml --img-size 768 --batch 8 --device 0 --tta
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
