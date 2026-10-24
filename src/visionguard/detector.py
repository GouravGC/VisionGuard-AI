
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from visionguard.config import (
    ANALYZED_CLASS_IDS,
    CONFIDENCE_THRESHOLD,
    IMAGE_SIZE,
    MODEL_PATH,
)


class VisionGuardDetector:
    """YOLO-based visual detector for helmet and head evidence."""

    def __init__(self, model_path: str | Path = MODEL_PATH):
        self.model_path = Path(model_path)

        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Model weights not found: {self.model_path}. "
                "Place visionguard_helmet.pt in artifacts/models "
                "or set VISIONGUARD_MODEL_PATH."
            )

        self.model = YOLO(str(self.model_path))

    def detect(self, image: np.ndarray) -> list[dict]:
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError("Input image is empty or invalid.")

        results = self.model.predict(
            source=image,
            conf=CONFIDENCE_THRESHOLD,
            imgsz=IMAGE_SIZE,
            verbose=False,
        )

        detections = []
        if not results or results[0].boxes is None:
            return detections

        for box in results[0].boxes:
            class_id = int(box.cls.item())

            if class_id not in ANALYZED_CLASS_IDS:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            detections.append(
                {
                    "class_id": class_id,
                    "label": str(self.model.names[class_id]),
                    "confidence": round(float(box.conf.item()), 4),
                    "bbox": [x1, y1, x2, y2],
                }
            )

        return detections
