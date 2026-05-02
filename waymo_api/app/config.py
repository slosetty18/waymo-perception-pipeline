from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    model_path: str = "models/waymo_yolov8m_distilled_best.pt"
    model_name: str = "YOLOv8m Ensemble Distilled"
    confidence_threshold: float = 0.35
    app_version: str = "1.0.0"

settings = Settings()
