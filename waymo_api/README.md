# Waymo Perception API

Object detection REST API powered by an ensemble-distilled YOLOv8m model trained on the Waymo Open Dataset.

## Model Performance

| Metric | Value |
|--------|-------|
| mAP@0.5 | 0.866 |
| mAP@0.5:0.95 | 0.629 |
| Inference Speed | 4.8ms |

## Classes Detected

- Vehicle
- Pedestrian
- Cyclist
- Sign

## Project Structure
```
waymo_api/
├── app/
│   ├── main.py          # FastAPI app with lifespan
│   ├── config.py        # Settings
│   ├── schemas/
│   │   └── inference.py # Request/response schemas
│   ├── services/
│   │   └── model.py     # YOLOv8m model service
│   └── api/
│       └── endpoints.py # Health + predict endpoints
├── models/              # Model weights
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```
## Run Locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Visit http://localhost:8000/docs for Swagger UI.

## Run with Docker

```bash
docker-compose up --build
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /api/v1/health | Health check |
| POST | /api/v1/predict | Run object detection |

## Example Request

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -F "file=@your_image.jpg"
```

## Example Response

```json
{
  "detections": [
    {
      "class_id": 0,
      "class_name": "Vehicle",
      "confidence": 0.92,
      "bbox": {"x1": 100.0, "y1": 200.0, "x2": 300.0, "y2": 400.0}
    }
  ],
  "count": 1,
  "image_size": {"width": 640, "height": 480}
}
```

## License

This project uses the Waymo Open Dataset — non-commercial use only.

## Observations

### Confidence Threshold
- Default threshold: 0.35 (production)
- Model returns only high-confidence detections at 0.35
- Lowering to 0.05 returns more detections including pedestrians

### Domain Shift
- Model trained on Waymo dashcam images (roof-mounted camera, forward-facing)
- Vehicle detection works well on out-of-domain images (confidence 0.84)
- Pedestrian detection confidence drops on non-Waymo images due to:
  - Different camera angle
  - Different lighting conditions
  - Different image perspective
- Pedestrians detected at confidence 0.054 on street-level photos
  vs 0.797 recall on Waymo images

### Per-Class Performance (Best Model)
| Class | Precision | Recall | mAP@0.5 |
|-------|-----------|--------|---------|
| Vehicle | 0.942 | 0.849 | 0.908 |
| Pedestrian | 0.912 | 0.797 | 0.880 |
| Cyclist | 0.871 | 0.745 | 0.808 |
