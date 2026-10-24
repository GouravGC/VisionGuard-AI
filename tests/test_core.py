
import numpy as np
import pytest

from visionguard.analyzer import analyze_detections, annotate_image


def test_head_without_helmet_requests_human_review():
    result = analyze_detections([
        {
            "class_id": 1,
            "label": "head",
            "confidence": 0.91,
            "bbox": [10, 10, 60, 70],
        }
    ])

    assert result["status"] == "REVIEW_REQUIRED"
    assert result["human_review_required"] is True
    assert result["head_count"] == 1
    assert result["helmet_count"] == 0


def test_no_detections_does_not_claim_scene_is_safe():
    result = analyze_detections([])

    assert result["status"] == "INSUFFICIENT_EVIDENCE"
    assert result["human_review_required"] is True


def test_person_class_is_not_counted_as_ppe_evidence():
    result = analyze_detections([
        {
            "class_id": 2,
            "label": "person",
            "confidence": 0.99,
            "bbox": [1, 2, 30, 50],
        }
    ])

    assert result["helmet_count"] == 0
    assert result["head_count"] == 0
    assert result["status"] == "INSUFFICIENT_EVIDENCE"


def test_invalid_image_is_rejected():
    with pytest.raises(ValueError):
        annotate_image(np.array([]), [])
