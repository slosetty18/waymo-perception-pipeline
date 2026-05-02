from ultralytics import YOLO
from PIL import Image
from app.config import settings

CLASS_NAMES = ["Vehicle", "Pedestrian", "Sign", "Cyclist"]

class WaymoModelService:
    def __init__(self):
        self.model = None
        self._initialized = False

    def initialize(self):
        """Load the YOLOv8m Distilled model."""
        if self._initialized:
            return
        print(f"Loading model from {settings.model_path}...")
        self.model = YOLO(settings.model_path)
        self._initialized = True
        print("Model loaded successfully ✅")

    def predict(self, image: Image.Image):
        """Run inference on a PIL image."""
        self.initialize()

        results = self.model.predict(
            image,
            conf=settings.confidence_threshold,
            verbose=False
        )[0]

        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append({
                "class_id": int(box.cls),
                "class_name": CLASS_NAMES[int(box.cls)],
                "confidence": round(float(box.conf), 3),
                "bbox": {
                    "x1": round(x1, 1),
                    "y1": round(y1, 1),
                    "x2": round(x2, 1),
                    "y2": round(y2, 1)
                }
            })

        return detections, image.width, image.height

# Global instance
_model_service = None

def get_model_service():
    global _model_service
    if _model_service is None:
        _model_service = WaymoModelService()
    return _model_service
