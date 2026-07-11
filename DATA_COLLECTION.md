# 路面标线缺失数据采集与增量标注规范

## 1. 采集目标

采集对象是能够从道路上下文判断的标线缺失、断裂或严重磨损区段。每个采集批次同时保留正常道路负样本，避免模型把所有路面纹理都预测为缺失。

建议最终数据至少包含：

| 条件 | 建议占比 |
|---|---:|
| 明显缺失/断裂/严重磨损正样本 | 60%-75% |
| 完好标线、裂缝、坑槽等负样本 | 25%-40% |
| 夜间、黄昏、阴影、逆光 | 不少于 20% |
| 雨天或湿路面 | 不少于 10% |
| 城市道路、乡村道路、高速道路 | 每类不少于 15% |
| 弯道、坡道、远距离小目标 | 不少于 20% |

## 2. 拍摄要求

- 图片分辨率不低于 1280x720；工具的最低接收阈值为 640x480。
- 避免连续保存几乎相同的视频帧；从视频抽帧时建议不超过 1 FPS。
- 同一次行驶、同一路段或同一视频放在一个独立子目录中，导出时会作为一个采集组整体划入 train 或 val。
- 保留夜间和雨天图，不要仅因亮度低而删除；严重失焦、镜头完全遮挡和无法辨认路面的图片应剔除。
- 避免采集可识别的人脸和车牌；必要时先脱敏。

推荐目录：

```text
incoming_data/
├── 20260711_cityA_day_sunny_session01/
├── 20260711_cityA_night_dry_session02/
└── 20260712_highway_rain_session03/
```

## 3. 导入新采集批次

```bash
python annotate.py ingest \
  --source incoming_data/20260711_cityA_day_sunny_session01 \
  --batch 20260711_cityA_session01 \
  --scene urban \
  --weather sunny \
  --time-of-day day
```

导入会执行 SHA-256 去重、分辨率检查、亮度和清晰度统计，并把可用图片复制到 `annotations/road_mark_missing/source_images/`。拒绝项记录在 `annotations/road_mark_missing/ingest_rejected.csv`。

查看进度：

```bash
python annotate.py status
```

## 4. 标注规则

- 只框缺失或严重磨损部分，不框整条仍然完好的标线。
- 框应紧贴目标，保留少量上下文，不要覆盖大片无关路面。
- 裂缝、坑槽、修补、阴影和车辆遮挡不是路面标线缺失。
- 无目标图片按 `N` 确认为负样本，不能直接跳过。
- 模糊到无法确定是否缺失的图片不应作为正样本。

使用内置稳定标注器人工标注：

```bash
python annotate.py review
```

使用鼠标左键画框、`S` 保存正样本、`N` 确认负样本、`A/D` 切换。程序会直接保存 YOLO 标签和复核状态。

LabelImg 仅作为可选兼容工具：

```bash
python annotate.py review-labelimg
```

只有在 LabelImg 中逐张检查全部剩余图片后，才运行 `python annotate.py labelimg-sync --accept-unlabeled-negative`。

有第一版单类模型后，先生成预标注再人工复核：

```bash
python annotate.py prelabel --weights runs/train/exp_baseline_ciou/weights/best.pt --device 0
python annotate.py review
```

预标注状态不等于真值，未经 `S` 或 `N` 人工确认的图片不会导出。

## 5. 导出与训练

```bash
python annotate.py export --output "new data1" --data data/road_mark_missing.yaml --train-ratio 0.8 --force
python experiment.py --exp EXP-00 --data data/road_mark_missing.yaml --outputs runs/roadmark_missing_audit
python experiment.py --exp EXP-01 --data data/road_mark_missing.yaml --device 0 --profile accuracy
```

导出按采集组划分 train/val，避免同一路段或相邻视频帧跨集合造成指标虚高。正式实验前建议至少 300 张正样本和 100 张负样本；论文主实验应继续扩充到 1000 张以上正样本，并进行第二人抽检。
