# waymo-perception-pipeline

## Results

| Metric | Score |
|--------|-------|
| mAP@0.5 | 0.461 |
| mAP@0.5:0.95 | 0.274 |
| Precision | 0.808 |
| Recall | 0.364 |
| Inference Speed | 18.8ms per image |
| Training Data | 2 segments, FRONT camera only |

## Notebooks

| Notebook | Description |
|----------|-------------|
| `01_data_exploration_and_tracking.ipynb` | Data parsing, visualization, object tracking, motion prediction |
| `02_yolov8_object_detection_training.ipynb` | YOLOv8n training, evaluation, and model export to GCS |

## Tech Stack

Python · TensorFlow · YOLOv8 · Google Cloud Storage · Pandas · NumPy · Matplotlib

## Dataset

[Waymo Open Dataset v1.4.2](https://waymo.com/open/) — 798 driving segments, 5 synchronized cameras, ground-truth annotations.
