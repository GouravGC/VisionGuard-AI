
import cv2
import numpy as np

from visionguard.config import CLASS_NAMES


def analyze_detections(detections: list[dict]) -> dict:
    """Convert visual detections into cautious, reviewable findings."""

    helmets = [d for d in detections if d["class_id"] == 0]
    heads = [d for d in detections if d["class_id"] == 1]

    # These are image-level indicators, not proof that a particular
    # worker is wearing or missing PPE.
    review_required = bool(heads)

    if heads and not helmets:
        status = "REVIEW_REQUIRED"
        message = (
            "Head-like regions detected without a detected helmet. "
            "Request human review; this is not a confirmed violation."
        )
    elif heads and helmets:
        status = "POSSIBLE_PPE_PRESENT"
        message = (
            "Both head-like regions and helmet detections were found. "
            "Confirm PPE compliance through human review."
        )
    elif helmets:
        status = "HELMET_DETECTED"
        message = (
            "Helmet-like objects detected. Confirm context and correct "
            "wearing position through human review."
        )
    else:
        status = "INSUFFICIENT_EVIDENCE"
        message = (
            "No helmet or head detections passed the confidence threshold. "
            "This does not establish that the scene is safe."
        )

    return {
        "status": status,
        "message": message,
        "helmet_count": len(helmets),
        "head_count": len(heads),
        "detection_count": len(detections),
        "human_review_required": review_required or status == "INSUFFICIENT_EVIDENCE",
        "detections": detections,
    }


def annotate_image(image: np.ndarray, detections: list[dict]) -> np.ndarray:
    """Draw detected boxes and labels using OpenCV."""

    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("Input image is empty or invalid.")

    output = image.copy()

    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        label = detection.get("label", CLASS_NAMES.get(detection["class_id"], "object"))
        confidence = float(detection.get("confidence", 0.0))
        color = (0, 180, 0) if detection["class_id"] == 0 else (0, 140, 255)

        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            output,
            f"{label} {confidence:.2f}",
            (x1, max(y1 - 8, 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    return output
