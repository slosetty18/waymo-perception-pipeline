from fastapi import APIRouter, File, UploadFile, HTTPException
from PIL import Image
import io
from app.schemas.inference import PredictionResponse, HealthResponse
from app.services.model import get_model_service
from app.config import settings

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if the API is running."""
    return {
        "status": "healthy",
        "model": settings.model_name,
        "version": settings.app_version,
        "mAP50": 0.866
    }

@router.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Run object detection on an uploaded image.

    - **file**: Image file (jpg, png, etc.)
    - **returns**: List of detections with class, confidence and bounding box
    """
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        model_service = get_model_service()
        detections, width, height = model_service.predict(image)

        return PredictionResponse(
            detections=detections,
            count=len(detections),
            image_size={"width": width, "height": height}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
