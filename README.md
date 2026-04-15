# waymo-perception-pipeline

## Results

### Comparison vs All Experiments

| Metric | Baseline YOLOv8n | Exp 1 YOLOv8m | Exp 2 Replay | Exp 3 YOLO26x 150seg | Exp 4 RF-DETR-L 150seg |
|--------|-----------------|---------------|--------------|----------------------|------------------------|
| mAP@0.5 | 0.461 | 0.736 | 0.687 | 0.779 | **0.826** |
| mAP@0.5:0.95 | 0.274 | 0.478 | 0.433 | 0.510 | **0.477** |
| Precision | 0.808 | 0.925 | 0.857 | 0.892 | **0.876** |
| Recall | 0.364 | 0.630 | 0.607 | 0.687 | **0.750** |
| Pedestrian Recall | 0.0 | 0.624 | 0.604 | 0.679 | **0.748** |
| Cyclist Recall | 0.0 | 0.551 | 0.515 | 0.622 | **0.689** |
| Inference Speed | 18.8ms | 4.8ms | 3.8ms | 8.9ms | TBD |

### Per-Class Performance — Best Model (Exp 4 RF-DETR-L, 150 segments)

| Class | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|-------|-----------|--------|---------|--------------|
| Vehicle | 0.903 | 0.813 | 0.875 | 0.572 |
| Pedestrian | 0.851 | 0.748 | 0.817 | 0.431 |
| Sign | — | — | — | — |
| Cyclist | 0.872 | 0.689 | 0.784 | 0.429 |

### Key Learnings
- **Exp 2:** Experience replay with 10% old data insufficient — mAP dropped from 0.736 to 0.687
- **Exp 3:** YOLO26x on 150 segments beats all previous models — mAP 0.779, Cyclist recall +7.1%
- **Exp 4:** RF-DETR-L beats YOLO26x by +4.7% mAP@0.5 (0.826 vs 0.779) — transformer architecture excels at pedestrian and cyclist detection
- **Engineering:** GCS checkpoint callbacks + resume logic essential for long training runs on Colab Pro+
- **Engineering:** RF-DETR requires `num_workers=0` on Colab to prevent dataloader deadlock

## Notebooks

| Notebook | Description |
|----------|-------------|
| `waymo_01_data_exploration_and_tracking.ipynb` | Data parsing, visualization, object tracking, motion prediction, linear regression |
| `waymo_02_yolov8_object_detection_training.ipynb` | Baseline — YOLOv8n training on 2 segments |
| `waymo_03_yolov8m_50segment_training.ipynb` | Exp 1 — YOLOv8m training on 50 segments — extraction pipeline, label conversion, full training |
| `waymo_04_yolov8m_experience_replay.ipynb` | Exp 2 — YOLOv8m experience replay on 105 segments — incremental learning experiment |
| `waymo_05_yolo26x_150segment_training.ipynb` | Exp 3 — YOLO26x training on 150 segments — CNN+Attention architecture |
| `waymo_06_rfdetr_150segment_training.ipynb` | Exp 4 — RF-DETR-L training on 150 segments — transformer-based detection |
| `waymo_07_ensemble.ipynb` | Exp 5 — RF-DETR-L + YOLO26x Ensemble (upcoming) |

## Experiment Plan

| Exp | NB | Model | Data | mAP@0.5 | Status |
|-----|-----|-------|------|---------|--------|
| Baseline | NB2 | YOLOv8n | 2 segments | 0.461 | ✅ Done |
| Exp 1 | NB3 | YOLOv8m | 50 segments | 0.736 | ✅ Done |
| Exp 2 | NB4 | YOLOv8m Experience Replay | 105 segments | 0.687 | ✅ Done |
| Exp 3 | NB5 | YOLO26x | 150 segments | 0.779 | ✅ Done |
| Exp 4 | NB6 | RF-DETR-L | 150 segments | **0.826** | ✅ Done |
| Exp 5 | NB7 | RF-DETR-L + YOLO26x Ensemble | 150 segments | TBD | 🔲 Next |
| Exp 6 | NB8 | Deployment | — | — | 🔲 Planned |

## Tech Stack
Python · YOLOv8 · YOLO26x · RF-DETR · Google Cloud Storage · Colab Pro+ · Pandas · NumPy · Matplotlib

## Dataset
[Waymo Open Dataset v1.4.2](https://waymo.com/open/) — 798 driving
segments, 5 synchronized cameras (FRONT camera used), ground-truth
annotations. 150 segments extracted (~29,700 images).
