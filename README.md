# waymo-perception-pipeline

## Results

### Latest Model — YOLOv8m (50 segments)
| Metric | YOLOv8n (2 segments) | YOLOv8m (50 segments) | Improvement |
|--------|---------------------|----------------------|-------------|
| mAP@0.5 | 0.461 | **0.736** | +59% |
| mAP@0.5:0.95 | 0.274 | **0.478** | +74% |
| Precision | 0.808 | **0.925** | +14% |
| Recall | 0.364 | **0.630** | +73% |
| Inference Speed | 18.8ms | **4.8ms** | 4x faster |

### Per-Class Performance (YOLOv8m)
| Class | Precision | Recall | mAP@0.5 |
|-------|-----------|--------|---------|
| Vehicle | 0.929 | 0.715 | 0.801 |
| Pedestrian | 0.897 | 0.624 | 0.727 |
| Sign | — | — | — |
| Cyclist | 0.949 | 0.551 | 0.681 |

## Notebooks
| Notebook | Description |
|----------|-------------|
| `01_data_exploration_and_tracking.ipynb` | Data parsing, visualization, object tracking, motion prediction |
| `02_yolov8_object_detection_training.ipynb` | YOLOv8n baseline training on 2 segments |
| `03_yolov8m_50segment_training.ipynb` | YOLOv8m training on 50 segments — extraction pipeline, label conversion, full training |

## Tech Stack
Python · TensorFlow · YOLOv8 · Google Cloud Storage · Pandas · NumPy · Matplotlib

## Dataset
[Waymo Open Dataset v1.4.2](https://waymo.com/open/) — 798 driving segments, 5 synchronized cameras, ground-truth annotations. 50 segments extracted for training (~9,900 images).



