# waymo-perception-pipeline

[ML Blog Series on Medium](https://medium.com/@slosetty18)

## Results

### Comparison vs All Experiments

| Metric | Baseline YOLOv8n | Exp 1 YOLOv8m | Exp 2 Replay | Exp 3 YOLO26x 150seg | Exp 4 RF-DETR-L 150seg | Exp 5 YOLOv8m Ensemble Distilled |
|--------|-----------------|---------------|--------------|----------------------|------------------------|----------------------------------|
| mAP@0.5 | 0.461 | 0.736 | 0.687 | 0.779 | 0.826 | **0.866** |
| mAP@0.5:0.95 | 0.274 | 0.478 | 0.433 | 0.510 | 0.477 | **0.629** |
| Precision | 0.808 | 0.925 | 0.857 | 0.892 | 0.876 | **0.908** |
| Recall | 0.364 | 0.630 | 0.607 | 0.687 | 0.750 | **0.797** |
| Pedestrian Recall | 0.0 | 0.624 | 0.604 | 0.679 | 0.748 | **0.797** |
| Cyclist Recall | 0.0 | 0.551 | 0.515 | 0.622 | 0.689 | **0.745** |
| Inference Speed | 18.8ms | 4.8ms | 3.8ms | 8.9ms | ~50ms | **4.8ms** |

### Per-Class Performance — Best Model (Exp 5 YOLOv8m Ensemble Distilled)

| Class | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|-------|-----------|--------|---------|--------------|
| Vehicle | 0.942 | 0.849 | 0.908 | 0.719 |
| Pedestrian | 0.912 | 0.797 | 0.880 | 0.627 |
| Sign | — | — | — | — |
| Cyclist | 0.871 | 0.745 | 0.808 | 0.539 |

### Key Learnings
- **Exp 2:** Experience replay with 10% old data insufficient — mAP dropped from 0.736 to 0.687
- **Exp 3:** YOLO26x on 150 segments beats all previous models — mAP 0.779, Cyclist recall +7.1%
- **Exp 4:** RF-DETR-L beats YOLO26x by +4.7% mAP@0.5 (0.826 vs 0.779) — transformer architecture excels at pedestrian and cyclist detection
- **Exp 5:** Ensemble distillation (RF-DETR-L + YOLO26x → YOLOv8m) achieves best mAP@0.5 of 0.866 at only 4.8ms inference — beats RF-DETR-L by +4.0% while being 10x faster
- **Engineering:** GCS checkpoint callbacks + resume logic essential for long training runs on Colab Pro+
- **Engineering:** RF-DETR requires `num_workers=0` on Colab to prevent dataloader deadlock
- **Engineering:** `caffeinate` command on Mac prevents sleep during long training runs
- **Engineering:** Docker build requires `--platform linux/amd64` on M2 Mac for Cloud Run deployment
  ## Observations
- Model detects vehicles confidently (0.845) even on photos it has never seen before
- Pedestrians are harder to detect on regular phone photos — model was trained on car roof camera, not street-level photos
- Lowering confidence threshold from 0.35 to 0.05 finds more objects but also more false detections
- Tested on real street photos from Phoenix and Nigeria — vehicles detected reliably in both

  ## Known Limitations & Production Gaps

- **Sign recall not measured** — Sign class annotations exist but model shows 0 recall; not production-safe
- **Pedestrian & Cyclist recall at 0 in baseline** — improved through iterations but still not at production threshold
- **Front camera only** — production AV systems use 5 cameras + LiDAR fusion for 360° perception
- **150 of 798 training segments used** — split 80% train / 20% val (~23,760 train / ~5,940 val images); training on full 798 segments may significantly improve recall on underrepresented classes
- **Adverse weather & night condition coverage not verified** — model behavior in adverse conditions untested
- **US cities only** — dataset collected across specific US locations; generalization to other countries, road layouts, and signage not validated
- **Not validated on dashcam or street-level images** — model trained on roof-mounted camera at ~1920×1280; performance degrades on phone or street-level photos
- **Inference tested on Colab GPU** — real-world edge deployment (Jetson, in-vehicle compute) not benchmarked

### Live Demo
 Platform | URL | Purpose |
|---|---|---|
| Ultralytics Platform | https://platform.ultralytics.com/ | Visual demo — upload image, see detections ✅ |
| Ultralytics Inference API | https://predict-69e551e576657ed89ece-dproatj77a-wn.a.run.app | Ultralytics hosted inference endpoint ✅ |
| Cloud Run API | https://waymo-perception-725477696855.us-central1.run.app | Production REST API ✅ |
### API Usage
```bash
# Health check
curl https://waymo-perception-725477696855.us-central1.run.app/

# Detect objects in image
curl -X POST \
  https://waymo-perception-725477696855.us-central1.run.app/detect \
  -F "file=@your_image.jpg"
```

## Notebooks

| Notebook | Description |
|----------|-------------|
| `waymo_01_data_exploration_and_tracking.ipynb` | Data parsing, visualization, object tracking, motion prediction, linear regression |
| `waymo_02_yolov8_object_detection_training.ipynb` | Baseline — YOLOv8n training on 2 segments |
| `waymo_03_yolov8m_50segment_training.ipynb` | Exp 1 — YOLOv8m training on 50 segments |
| `waymo_04_yolov8m_experience_replay.ipynb` | Exp 2 — YOLOv8m experience replay on 105 segments |
| `waymo_05_yolo26x_150segment_training.ipynb` | Exp 3 — YOLO26x training on 150 segments |
| `waymo_06_rfdetr_150segment_training.ipynb` | Exp 4 — RF-DETR-L training on 150 segments |
| `waymo_07_ensemble_distillation.ipynb` | Exp 5 — Ensemble distillation (RF-DETR-L + YOLO26x → YOLOv8m) |
| `waymo_08_Deployment.ipynb` | Exp 6 — FastAPI + Docker + Cloud Run + Ultralytics HUB |

## Experiment Plan

| Exp | NB | Model | Data | mAP@0.5 | Status |
|-----|-----|-------|------|---------|--------|
| Baseline | NB2 | YOLOv8n | 2 segments | 0.461 | ✅ Done |
| Exp 1 | NB3 | YOLOv8m | 50 segments | 0.736 | ✅ Done |
| Exp 2 | NB4 | YOLOv8m Experience Replay | 105 segments | 0.687 | ✅ Done |
| Exp 3 | NB5 | YOLO26x | 150 segments | 0.779 | ✅ Done |
| Exp 4 | NB6 | RF-DETR-L | 150 segments | 0.826 | ✅ Done |
| Exp 5 | NB7 | YOLOv8m Ensemble Distilled | 150 segments | **0.866** | ✅ Done |
| Exp 6 | NB8 | Deployment | — | — | ✅ Done |

## Tech Stack
Python · YOLOv8 · YOLO26x · RF-DETR · Ensemble Distillation · WBF · FastAPI · Docker · Google Cloud Run · Ultralytics HUB · Google Cloud Storage · Colab Pro+ · Pandas · NumPy · Matplotlib

## Dataset
[Waymo Open Dataset v1.4.2](https://waymo.com/open/) — 798 driving segments, 5 synchronized cameras (FRONT camera used), ground-truth annotations. 150 segments extracted (~29,700 images).

## License
This project uses the Waymo Open Dataset, licensed for **non-commercial use only** under the [Waymo Dataset License Agreement](https://waymo.com/open/terms/). Any models trained on this dataset may not be used for commercial purposes or deployed in real vehicles.
