from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.api.endpoints import router
from app.services.model import get_model_service
from app.config import settings

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load the YOLOv8m detection model before accepting requests."""
    print("Loading Waymo perception model...")
    service = get_model_service()
    service.initialize()
    print("Model loaded and ready.")
    yield

app = FastAPI(
    title="Waymo Perception API",
    description="""
Object detection API powered by an ensemble-distilled YOLOv8m model
trained on the Waymo Open Dataset.

**Classes detected:** Vehicle, Pedestrian, Cyclist, Sign

**Model performance:** mAP@0.5 = 0.866 | Inference = 4.8ms

License: Waymo Open Dataset — non-commercial use only.
    """,
    version=settings.app_version,
    lifespan=lifespan,
)

app.include_router(router, prefix="/api/v1", tags=["inference"])
