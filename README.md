# waymo-perception-pipeline

## Results

### Experiment Comparison
| Metric | YOLOv8n (2 seg) | YOLOv8m (50 seg) | YOLOv8m Replay (105 seg) | YOLO26x (150 seg) |
|--------|----------------|-----------------|--------------------------|-------------------|
| mAP@0.5 | 0.461 | 0.736 | 0.687 | **0.779** |
| mAP@0.5:0.95 | 0.274 | 0.478 | 0.433 | **0.510** |
| Precision | 0.808 | 0.925 | 0.857 | **0.892** |
| Recall | 0.364 | 0.630 | 0.607 | **0.687** |
| Inference Speed | 18.8ms | 4.8ms | 3.8ms | 8.9ms |

### Per-Class Performance — Best Model (YOLO26x, 150 segments)
| Class | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|-------|-----------|--------|---------|--------------|
| Vehicle | 0.921 | 0.761 | 0.845 | 0.618 |
| Pedestrian | 0.886 | 0.679 | 0.782 | 0.483 |
| Sign | — | — | — | — |
| Cyclist | 0.868 | 0.622 | 0.711 | 0.429 |

### Key Learnings
- **Exp 2:** Experience replay with 10% old data insufficient — mAP dropped from 0.736 to 0.687
- **Exp 3:** YOLO26x on 150 segments beats all previous models — mAP 0.779, Cyclist recall +7.1%
- **Engineering:** GCS checkpoint callbacks essential for long training runs on Colab

## Notebooks
| Notebook | Description |
|----------|-------------|
| `waymo_01_data_exploration_and_tracking.ipynb` | Data parsing, visualization, object tracking, motion prediction |
| `waymo_02_yolov8_object_detection_training.ipynb` | YOLOv8n baseline training on 2 segments |
| `waymo_03_yolov8m_50segment_training.ipynb` | YOLOv8m training on 50 segments — extraction pipeline, label conversion, full training |
| `waymo_04_yolov8m_experience_replay.ipynb` | YOLOv8m experience replay on 105 segments — incremental learning experiment |
| `waymo_05_yolo26x_150segment_training.ipynb` | YOLO26x training on 150 segments — CNN+Attention architecture |

## Experiment Plan
| Exp | Model | Data | mAP@0.5 | Status |
|-----|-------|------|---------|--------|
| Baseline | YOLOv8n | 2 segments | 0.461 | ✅ Done |
| Exp 1 | YOLOv8m | 50 segments | 0.736 | ✅ Done |
| Exp 2 | YOLOv8m Experience Replay | 105 segments | 0.687 | ✅ Done |
| Exp 3 | YOLO26x | 150 segments | 0.779 | ✅ Done |
| Exp 4 | RF-DETR-L | 150 segments | TBD | 🔲 Next |
| Exp 5 | RF-DETR-L + YOLO26x Ensemble | 150 segments | TBD | 🔲 Planned |

## Tech Stack
Python · TensorFlow · YOLOv8 · YOLO26x · Google Cloud Storage · Pandas · NumPy · Matplotlib

## Dataset
[Waymo Open Dataset v1.4.2](https://waymo.com/open/) — 798 driving 
segments, 5 synchronized cameras (FRONT camera used), ground-truth 
annotations. 150 segments extracted (~29,700 images).
