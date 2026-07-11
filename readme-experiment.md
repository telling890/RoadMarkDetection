# 路面标线缺失检测实验 README

本文档用于执行“路面标线缺失检测”实验。实验目标不是普通道路目标检测，而是识别道路标线缺失、断裂、磨损、遮挡导致不可见等异常区域，并验证改进 YOLO26 在复杂道路环境下的检测效果。

实验主入口：

```bash
python experiment.py --device 0
```

其中 `--device 0` 表示使用第 1 张 CUDA GPU。

## 0. 实验定位

本实验围绕一个核心问题展开：

> 在夜间、阴影、雨天、磨损、遮挡等复杂道路环境下，改进 YOLO26 是否能更准确、更稳定地检测路面标线缺失区域？

回答该问题的前提是使用经过人工确认的路面标线缺失真值。当前仓库中的 `D00-D90` 标签源自道路病害检测，只能验证代码链路；在完成重标注或逐图复核前，不得将现有结果解释为路面标线缺失检测精度。

本文档将实验拆成 10 个可执行任务：

| 实验 | 名称 | 目标 |
|---|---|---|
| EXP-00 | 数据集与类别协议 | 确认缺失标线数据、标签和类别编号可训练 |
| EXP-01 | YOLO26 Clean Baseline | 建立缺失检测基线 |
| EXP-02 | 复杂环境鲁棒性 | 评估夜间、雨天、阴影、磨损、遮挡 |
| EXP-03 | EMA 注意力模块 | 提升低对比度缺失区域特征表达 |
| EXP-04 | BiFPN 多尺度融合 | 提升小尺度、远距离缺失目标检测 |
| EXP-05 | C2f-DCN 形变建模 | 提升不规则缺失边界建模能力 |
| EXP-06 | Wise-IoU 定位损失 | 提升缺失区域边界框定位质量 |
| EXP-07 | 完整模型主对比 | 汇总 baseline 和消融实验 |
| EXP-08 | 实时推理验证 | 验证图片、视频、摄像头输入 |
| EXP-09 | PyQt5 Demo | 验证可视化展示系统 |

## 1. GPU 环境检查

训练前先确认 CUDA 可用：

```bash
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

当前已验证：

```text
torch: 2.11.0+cu128
CUDA: 12.8
GPU: NVIDIA GeForce RTX 5070 Laptop GPU
```

若 `torch.cuda.is_available()` 输出 `False`，需要安装 CUDA 版 PyTorch。可参考 PyTorch 官方安装页：https://pytorch.org/get-started/locally/

## 2. 数据准备

数据配置：

```text
data/road_mark.yaml
```

训练目录：

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

目标实验类别协议暂定为：

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

类别显示名由 `data/road_mark.yaml` 控制，但显示名不会改变标注框的真实语义。数据语义状态记录在 `data/label_semantics.yaml`；正式实验要求 `verified_for_target_task: true`，并保留复核人、复核日期和标注规范版本。

从 `data/` 重新生成数据集：

```bash
python prepare_dataset.py --raw-data data --prepared-data dataset --data data/road_mark.yaml --outputs runs/experiments --force-selection --import-zips
```

如果原始 zip 不在 `data/`，但已解压缓存还在，可从缓存恢复：

```bash
python prepare_dataset.py --raw-data runs/_extracted_txt --prepared-data dataset --data data/road_mark.yaml --outputs runs/experiments --force-selection --import-zips
```

当前导入器支持：

- `data/**/*_txt.zip` 自动扫描。
- `runs/_extracted_txt` 这类已解压 Roboflow 目录。
- `labels` 目录标签。
- `labelTxt` 四点框标签。
- 四点框到 YOLO 水平框转换。
- 各 zip 类别编号到统一类别协议的映射。

当前数据审计结果：

| split | images | boxes | missing labels | invalid labels |
|---|---:|---:|---:|---:|
| train | 19302 | 64054 | 0 | 0 |
| val | 4127 | 13953 | 0 | 0 |

## 3. 快速开始

先执行 CPU 逻辑检查和 GPU 模型完整性检查：

```bash
python tests/smoke_test.py
python tests/integrity_test.py --require-gpu
```

数据审计：

```bash
python experiment.py --exp EXP-00 --outputs runs/experiments
```

1 轮 GPU 冒烟训练：

```bash
python experiment.py --exp EXP-01 --device 0 --epochs 1 --batch 8 --workers 4
```

正式 baseline：

```bash
python experiment.py --exp EXP-01 --device 0 --profile accuracy
```

完整消融：

```bash
python experiment.py --exp EXP-07 --device 0 --profile accuracy
```

`accuracy` 是默认正式训练配置：输入尺寸 768、batch 8、AdamW、cosine LR、multi-scale 和较温和的数据增强。RTX 5070 Laptop GPU 显存不足时先使用 `--batch 4`，保持所有对比组的输入尺寸和超参数一致。

查看实验清单：

```bash
python experiment.py --list
```

只预览命令：

```bash
python experiment.py --exp all --device 0 --dry-run
```

## 4. 实验执行顺序

建议按以下顺序推进：

| 阶段 | 实验 | 作用 | 主要产物 |
|---|---|---|---|
| Phase 0 | EXP-00 | 数据与标签可用性确认 | `dataset_audit.md` |
| Phase 1 | EXP-01 | 建立缺失检测 baseline | baseline 权重与验证指标 |
| Phase 2 | EXP-03, EXP-04, EXP-05 | 验证模块贡献 | 消融结果 |
| Phase 3 | EXP-06, EXP-07 | 验证损失函数和完整模型 | 主结果表与对比图 |
| Phase 4 | EXP-02 | 复杂环境鲁棒性补充 | 场景对比表 |
| Phase 5 | EXP-08, EXP-09 | 推理与展示验证 | 检测结果与 Demo 清单 |

最低闭环：

1. 完成真实路面标线缺失标注复核，并更新 `data/label_semantics.yaml`。
2. 跑 `EXP-00`，确认数据集无缺失标签、非法标签和语义错误。
3. 跑 `tests/integrity_test.py --require-gpu` 和 `EXP-01 --epochs 1`，确认 GPU 训练链路。
4. 跑完整 `EXP-01`，得到 baseline。
5. 跑 `EXP-07`，得到消融表。
6. 用独立测试集进行最终验证，再跑 `EXP-08` 输出可视化结果。

## 5. 实验任务

### EXP-00 数据集与类别协议

目的：固定标线缺失数据来源、目录结构、类别编号和 YOLO 标注格式。

命令：

```bash
python experiment.py --exp EXP-00 --outputs runs/experiments
```

如需重新导入：

```bash
python experiment.py --exp EXP-00 --raw-data data --prepared-data dataset --force-selection --outputs runs/experiments
```

产物：

```text
runs/experiments/dataset_stats.csv
runs/experiments/class_distribution.csv
runs/experiments/invalid_labels.csv
runs/experiments/missing_labels.json
runs/experiments/dataset_audit.md
```

完成标准：

- train/val 图片目录存在。
- 每张图片有对应标签。
- 标签为 YOLO 5 字段格式。
- 类别编号在 `0` 到 `9` 范围内。
- bbox 坐标已归一化。
- 抽样图像与类别名称的语义一致。
- `data/label_semantics.yaml` 已记录为人工验证通过。

### EXP-01 YOLO26 Clean Baseline

目的：建立路面标线缺失检测的基础性能参考。

命令：

```bash
python experiment.py --exp EXP-01 --device 0
```

底层训练命令：

```bash
python train.py --variant baseline --loss ciou --data data/road_mark.yaml --device 0
```

指标：

- Precision
- Recall
- F1
- mAP50
- mAP50-95
- FPS
- 各类别 Precision、Recall、F1、mAP50、mAP50-95

完成标准：

- 生成 `runs/train/exp_baseline_ciou/weights/best.pt`。
- 生成 `runs/val/exp_baseline_ciou/metrics.json`。
- baseline 可作为后续模块对比参考。

### EXP-02 复杂环境鲁棒性评估

目的：验证模型在夜间、阴影、雨天、模糊、磨损、遮挡等复杂场景中对标线缺失的检测能力。

命令：

```bash
python experiment.py --exp EXP-02 --device 0
```

如有单独场景验证集：

```bash
python experiment.py --exp EXP-02 --device 0 \
  --scenario-data night=data/night.yaml \
  --scenario-data rain=data/rain.yaml \
  --scenario-data shadow=data/shadow.yaml
```

产物：

```text
runs/experiments/complex_env_results.csv
```

完成标准：

- 至少包含 clean 与若干复杂场景。
- 完整模型在复杂场景中 Recall 或 mAP50 优于 baseline。
- clean 验证性能没有明显坍塌。

### EXP-03 EMA 注意力模块实验

目的：验证 EMA 是否提升低对比度、弱纹理、磨损缺失区域的特征表达。

命令：

```bash
python experiment.py --exp EXP-03 --device 0
```

对比：

- YOLO26
- YOLO26+EMA

完成标准：

- EMA 相比 baseline 在 Recall、mAP50 或低对比度样例中有可解释收益。

### EXP-04 BiFPN 多尺度融合实验

目的：验证 BiFPN 对远距离、小尺度、短段缺失标线的检测收益。

命令：

```bash
python experiment.py --exp EXP-04 --device 0
```

对比：

- YOLO26+EMA
- YOLO26+EMA+BiFPN

完成标准：

- 小目标或远距离缺失区域的召回更稳定。

### EXP-05 C2f-DCN 形变建模实验

目的：验证 C2f-DCN 对弯曲标线、透视变形标线和不规则缺失边界的建模能力。

命令：

```bash
python experiment.py --exp EXP-05 --device 0
```

对比：

- YOLO26+EMA+BiFPN
- YOLO26+C2f-DCN+EMA+BiFPN

完成标准：

- 不规则缺失边界、弯道、磨损断裂场景下 Recall 或 mAP50 提升。

### EXP-06 Wise-IoU 定位损失实验

目的：验证 Wise-IoU 是否改善缺失区域边界框定位质量。

命令：

```bash
python experiment.py --exp EXP-06 --device 0
```

对比：

- 完整结构 + CIoU
- 完整结构 + Wise-IoU

完成标准：

- Wise-IoU 对 mAP50-95 有提升，或减少明显框偏移案例。

### EXP-07 完整模型主对比

目的：汇总 baseline、各模块消融和完整模型对标线缺失检测的整体收益。

命令：

```bash
python experiment.py --exp EXP-07 --device 0
```

实验组：

| Model | 说明 |
|---|---|
| YOLO26 | clean baseline |
| YOLO26+EMA | 注意力增强 |
| YOLO26+EMA+BiFPN | 注意力 + 多尺度融合 |
| YOLO26+C2f-DCN+EMA+BiFPN | 完整结构 + CIoU |
| YOLO26+C2f-DCN+EMA+BiFPN+WiseIoU | 完整模型 |

产物：

```text
runs/experiments/experiment_results.csv
runs/experiments/experiment_results.md
runs/experiments/figures/figure4_ablation_metrics.png
```

完成标准：

- 完整模型在 Recall、mAP50 或 mAP50-95 上优于主要消融版本。
- FPS 仍满足巡检或演示需求。

### EXP-08 实时推理与视频验证

目的：验证图片、视频和摄像头下的路面标线缺失检测链路。

图片：

```bash
python experiment.py --exp EXP-08 --device 0 --weights runs/train/best.pt --source test.jpg
```

视频：

```bash
python experiment.py --exp EXP-08 --device 0 --weights runs/train/best.pt --source video.mp4
```

摄像头：

```bash
python experiment.py --exp EXP-08 --device 0 --weights runs/train/best.pt --source 0 --show
```

产物：

```text
runs/detect/roadmark_missing_detect/
runs/experiments/realtime_results.csv
```

### EXP-09 PyQt5 Demo 与结果展示

目的：验证可视化展示系统能加载模型并展示标线缺失检测结果。

生成检查清单：

```bash
python experiment.py --exp EXP-09
```

启动 GUI：

```bash
python experiment.py --exp EXP-09 --launch-gui
```

产物：

```text
runs/experiments/demo_checklist.md
```

## 6. 图表产物

| 图号 | 名称 | 来源 |
|---|---|---|
| Figure 1 | 标线缺失检测系统总体框架 | 方法设计 |
| Figure 2 | 改进 YOLO26 网络结构 | 模型 YAML 与模块说明 |
| Figure 3 | 复杂环境缺失检测可视化 | EXP-02, EXP-08 |
| Figure 4 | 消融实验指标对比图 | EXP-07 |
| Figure 5 | 实时推理或 Demo 截图 | EXP-08, EXP-09 |

## 7. 表格产物

| 表号 | 名称 | 来源 | 文件 |
|---|---|---|---|
| Table 1 | 路面标线缺失数据集统计 | EXP-00 | `dataset_stats.csv`, `class_distribution.csv` |
| Table 2 | 模型消融实验结果 | EXP-07 | `experiment_results.csv` |
| Table 3 | 复杂环境鲁棒性结果 | EXP-02 | `complex_env_results.csv` |
| Table 4 | 实时检测性能 | EXP-08 | `realtime_results.csv` |

## 8. 推荐命令组合

第一次检查：

```bash
python tests/smoke_test.py
python tests/integrity_test.py --require-gpu
python experiment.py --list
python experiment.py --exp all --device 0 --profile accuracy --dry-run
```

数据重新导入并审计：

```bash
python prepare_dataset.py --raw-data data --prepared-data dataset --data data/road_mark.yaml --outputs runs/experiments --force-selection --import-zips
python experiment.py --exp EXP-00 --outputs runs/experiments
```

正式训练：

```bash
python experiment.py --exp EXP-01 --device 0 --profile accuracy
python experiment.py --exp EXP-07 --device 0 --profile accuracy
```

独立验证：

```bash
python val.py --weights runs/train/exp_full_wise_iou/weights/best.pt --data data/road_mark.yaml --img-size 768 --batch 8 --device 0
python val.py --weights runs/train/exp_full_wise_iou/weights/best.pt --data data/road_mark.yaml --img-size 768 --batch 8 --device 0 --tta --name exp_full_wise_iou_tta
```

TTA 只用于最终精度评估，会降低 FPS。消融表必须全部关闭 TTA 或全部开启 TTA，不能混用。

推理展示：

```bash
python experiment.py --exp EXP-08 --device 0 --weights runs/train/best.pt --source test.jpg
python experiment.py --exp EXP-09 --launch-gui
```

## 9. Go / No-Go 判据

继续推进：

- 完整模型相对 baseline 的 Recall、mAP50 或 mAP50-95 有稳定提升。
- 复杂环境下缺失标线漏检减少。
- 小尺度、低对比度、遮挡缺失区域检测更稳定。
- FPS 满足实时或准实时展示。

需要调整：

- 完整模型只增加耗时，没有提升。
- Wise-IoU 造成定位指标下降。
- 复杂环境增强伤害 clean validation。
- GUI 或实时推理不稳定。

## 10. 注意事项

- 源数据 `D00-D90` 是道路病害标签，当前改名不是有效的路面标线缺失语义转换。
- 正式训练前必须重标注或人工复核，并补充“类别编号 - 中文含义 - 标注规则 - 正反样例”表。
- `other_marking_missing` 样本很少，训练和论文分析中应说明类别长尾问题。
- 1 轮训练只用于检查链路，不作为正式实验指标。
- 提高输入尺寸、增强和 TTA 只能优化已正确标注的数据，无法修复错误标签语义。
