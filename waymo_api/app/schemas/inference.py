from pydantic import BaseModel
from typing import List

class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox

class PredictionResponse(BaseModel):
    detections: List[Detection]
    count: int
    image_size: dict

    class Config:
        json_schema_extra = {
            "example": {
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
        }

class HealthResponse(BaseModel):
    status: str
    model: str
    version: str
    mAP50: float
