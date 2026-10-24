
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = Path(
    os.getenv(
        "VISIONGUARD_MODEL_PATH",
        str(PROJECT_ROOT / "artifacts" / "models" / "visionguard_helmet.pt"),
    )
)

CONFIDENCE_THRESHOLD = float(os.getenv("VISIONGUARD_CONFIDENCE", "0.35"))
IMAGE_SIZE = int(os.getenv("VISIONGUARD_IMAGE_SIZE", "416"))
MAX_VIDEO_FRAMES = int(os.getenv("VISIONGUARD_MAX_VIDEO_FRAMES", "30"))
VIDEO_FRAME_INTERVAL = int(os.getenv("VISIONGUARD_VIDEO_FRAME_INTERVAL", "15"))

# The trained model performed poorly on the person class.
# Analyze helmet/head detections only; do not claim reliable person detection.
CLASS_NAMES = {
    0: "helmet",
    1: "head",
    2: "person",
}

ANALYZED_CLASS_IDS = {0, 1}
