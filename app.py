
import json
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# Make the src-layout package importable when launched from the project root.
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from visionguard.analyzer import analyze_detections, annotate_image
from visionguard.config import (
    CONFIDENCE_THRESHOLD,
    IMAGE_SIZE,
    MAX_VIDEO_FRAMES,
    MODEL_PATH,
    VIDEO_FRAME_INTERVAL,
)
from visionguard.detector import VisionGuardDetector


# ---------------------------------------------------------
# Page setup
# ---------------------------------------------------------

st.set_page_config(
    page_title="VisionGuard AI | Inference Lab",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ VisionGuard AI")
st.caption(
    "OpenCV 5 · PPE visual inspection · "
    "PyTorch vs ONNX inference comparison"
)

st.warning(
    "Research prototype only. Detections may be incorrect or incomplete. "
    "This application is not a certified safety system. Human review is required."
)


# ---------------------------------------------------------
# Model discovery and runtime selection
# ---------------------------------------------------------

DEFAULT_MODEL = Path(MODEL_PATH)
MODELS_DIR = DEFAULT_MODEL.parent

PT_MODEL = MODELS_DIR / "visionguard_helmet.pt"
ONNX_MODEL = MODELS_DIR / "visionguard_helmet.onnx"

MODE_OPTIONS = [
    "PyTorch Native (.pt)",
    "ONNX (.onnx)",
    "Compare Both",
]


@st.cache_resource(show_spinner="Loading detection model...")
def load_detector(model_path: str):
    """Cache each model independently by its exact path."""
    return VisionGuardDetector(model_path)


with st.sidebar:
    st.header("⚙️ Inspection settings")

    inference_mode = st.selectbox(
        "Inference runtime",
        MODE_OPTIONS,
        index=0,
        help=(
            "Choose PyTorch, ONNX, or run both on the same inputs "
            "to compare results and observed inference time."
        ),
    )

    st.divider()
    st.subheader("Detection configuration")
    st.write(f"Confidence threshold: `{CONFIDENCE_THRESHOLD}`")
    st.write(f"Image size: `{IMAGE_SIZE}`")
    st.write(f"Video frame interval: `{VIDEO_FRAME_INTERVAL}`")
    st.write(f"Maximum sampled frames: `{MAX_VIDEO_FRAMES}`")

    st.divider()
    st.subheader("Model files")
    st.write(f"PyTorch: {'✅ Found' if PT_MODEL.is_file() else '❌ Missing'}")
    st.write(f"ONNX: {'✅ Found' if ONNX_MODEL.is_file() else '❌ Missing'}")

    st.info(
        "The person class is excluded because its validation performance "
        "was poor. Head and helmet detections are scene-level evidence, "
        "not proof of an individual worker's compliance."
    )


if inference_mode == "PyTorch Native (.pt)":
    selected_models = {"PyTorch": PT_MODEL}
elif inference_mode == "ONNX (.onnx)":
    selected_models = {"ONNX": ONNX_MODEL}
else:
    selected_models = {"PyTorch": PT_MODEL, "ONNX": ONNX_MODEL}


missing_models = [
    str(path)
    for path in selected_models.values()
    if not path.is_file()
]

if missing_models:
    st.error(
        "The selected runtime requires model file(s) that are missing:\n\n"
        + "\n".join(f"- `{path}`" for path in missing_models)
    )
    st.info(
        "Export the ONNX model from your trained PyTorch weights using:\n\n"
        "`python -c \"from ultralytics import YOLO; "
        "YOLO('artifacts/models/visionguard_helmet.pt').export("
        "format='onnx', imgsz=416)\"`"
    )
    st.stop()


try:
    detectors = {
        name: load_detector(str(path))
        for name, path in selected_models.items()
    }
except Exception as exc:
    st.error(f"Failed to load the selected model: {exc}")
    st.stop()


# ---------------------------------------------------------
# Inference and comparison helpers
# ---------------------------------------------------------

def run_inference(detector, frame):
    """Return detections, annotated image, report, and inference time."""
    start = time.perf_counter()
    detections = detector.detect(frame)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    report = analyze_detections(detections)
    annotated = annotate_image(frame, detections)

    return {
        "detections": detections,
        "annotated": annotated,
        "report": report,
        "inference_ms": round(elapsed_ms, 3),
    }


def box_iou(box_a, box_b):
    """Calculate intersection-over-union for two xyxy boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def compare_detections(detections_a, detections_b, iou_threshold=0.5):
    """
    Greedily match same-class boxes using IoU >= threshold.
    This measures agreement between models, not ground-truth accuracy.
    """
    unmatched_b = set(range(len(detections_b)))
    matched = 0

    for detection_a in detections_a:
        best_index = None
        best_iou = 0.0

        for index in unmatched_b:
            detection_b = detections_b[index]

            if detection_a["class_id"] != detection_b["class_id"]:
                continue

            overlap = box_iou(
                detection_a["bbox"],
                detection_b["bbox"],
            )

            if overlap > best_iou:
                best_iou = overlap
                best_index = index

        if best_index is not None and best_iou >= iou_threshold:
            matched += 1
            unmatched_b.remove(best_index)

    total_a = len(detections_a)
    total_b = len(detections_b)
    union_count = total_a + total_b - matched

    return {
        "matched_boxes": matched,
        "pytorch_detections": total_a,
        "onnx_detections": total_b,
        "agreement_percent": (
            round(100 * matched / union_count, 2)
            if union_count
            else 100.0
        ),
        "iou_threshold": iou_threshold,
    }


def display_result(name, result):
    """Display annotated image and review summary."""
    st.subheader(name)

    image_rgb = cv2.cvtColor(
        result["annotated"],
        cv2.COLOR_BGR2RGB,
    )
    st.image(image_rgb, use_container_width=True)

    report = result["report"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Helmets", report["helmet_count"])
    c2.metric("Heads", report["head_count"])
    c3.metric("Inference time", f"{result['inference_ms']:.2f} ms")

    st.write(f"**Status:** `{report['status']}`")
    st.write(report["message"])


def display_runtime_comparison(results):
    """Show runtime and detection comparisons for one input."""
    if len(results) != 2:
        return

    pt_result = results["PyTorch"]
    onnx_result = results["ONNX"]

    st.subheader("⚖️ Runtime comparison")

    pt_ms = pt_result["inference_ms"]
    onnx_ms = onnx_result["inference_ms"]

    c1, c2, c3 = st.columns(3)
    c1.metric("PyTorch", f"{pt_ms:.2f} ms")
    c2.metric("ONNX", f"{onnx_ms:.2f} ms")

    if onnx_ms > 0:
        speedup = pt_ms / onnx_ms
        if speedup >= 1:
            c3.metric("ONNX speed ratio", f"{speedup:.2f}×")
        else:
            c3.metric("PyTorch speed ratio", f"{1 / speedup:.2f}×")
    else:
        c3.metric("Speed ratio", "N/A")

    agreement = compare_detections(
        pt_result["detections"],
        onnx_result["detections"],
    )

    st.write("**Detection agreement**")
    st.json(agreement)

    st.caption(
        "Times measure the detector call only and may include warm-up "
        "or runtime initialization effects. Compare multiple runs for a "
        "fairer benchmark. Agreement is not accuracy; ground-truth labels "
        "are required to measure precision and recall."
    )


# ---------------------------------------------------------
# Input mode
# ---------------------------------------------------------

input_mode = st.radio(
    "Choose input type",
    ["Image inspection", "Video inspection"],
    horizontal=True,
)


# ---------------------------------------------------------
# Image inspection
# ---------------------------------------------------------

if input_mode == "Image inspection":
    uploaded = st.file_uploader(
        "Upload a workplace image",
        type=["jpg", "jpeg", "png", "webp"],
        key="image_upload",
    )

    if uploaded is not None:
        try:
            rgb = np.array(Image.open(uploaded).convert("RGB"))
            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            with st.spinner("Running selected inference runtime(s)..."):
                results = {
                    name: run_inference(detector, frame)
                    for name, detector in detectors.items()
                }

            if len(results) == 1:
                name, result = next(iter(results.items()))

                left, right = st.columns(2)
                with left:
                    st.subheader("Original image")
                    st.image(rgb, use_container_width=True)
                with right:
                    display_result(name, result)

                final_report = {
                    "application": "VisionGuard AI",
                    "input_type": "image",
                    "runtime": name,
                    "model_path": str(selected_models[name]),
                    "inference_ms": result["inference_ms"],
                    "result": result["report"],
                }

            else:
                original_col, pt_col, onnx_col = st.columns(3)

                with original_col:
                    st.subheader("Original image")
                    st.image(rgb, use_container_width=True)

                with pt_col:
                    display_result("PyTorch Native", results["PyTorch"])

                with onnx_col:
                    display_result("ONNX", results["ONNX"])

                display_runtime_comparison(results)

                final_report = {
                    "application": "VisionGuard AI",
                    "input_type": "image",
                    "comparison": {
                        name: {
                            "inference_ms": result["inference_ms"],
                            "result": result["report"],
                        }
                        for name, result in results.items()
                    },
                    "detection_agreement": compare_detections(
                        results["PyTorch"]["detections"],
                        results["ONNX"]["detections"],
                    ),
                }

            final_report["limitations"] = [
                "Research prototype; not a certified safety system.",
                "Head and helmet detections do not establish worker-level compliance.",
                "No detections do not prove a scene is safe.",
                "Runtime measurements are hardware- and warm-up-dependent.",
            ]

            st.download_button(
                "📥 Download inspection JSON",
                data=json.dumps(final_report, indent=2),
                file_name="visionguard_image_report.json",
                mime="application/json",
            )

        except Exception as exc:
            st.error(f"Image inference failed: {exc}")


# ---------------------------------------------------------
# Video inspection
# ---------------------------------------------------------

else:
    uploaded = st.file_uploader(
        "Upload a workplace video",
        type=["mp4", "mov", "avi", "mkv"],
        key="video_upload",
    )

    if uploaded is not None:
        st.caption(
            f"Frames are sampled every {VIDEO_FRAME_INTERVAL} frames, "
            f"up to {MAX_VIDEO_FRAMES} sampled frames."
        )

        if st.button("▶ Analyze video", type="primary"):
            temporary_path = None
            capture = None

            try:
                suffix = Path(uploaded.name).suffix or ".mp4"

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=suffix,
                ) as temp:
                    temp.write(uploaded.getvalue())
                    temporary_path = temp.name

                capture = cv2.VideoCapture(temporary_path)

                if not capture.isOpened():
                    raise ValueError("OpenCV could not open this video.")

                fps = capture.get(cv2.CAP_PROP_FPS)
                total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

                frame_step = max(1, VIDEO_FRAME_INTERVAL)
                sampled_count = 0
                frame_number = 0

                frame_records = []
                first_frame_results = None
                progress = st.progress(0.0)
                status = st.empty()

                while sampled_count < MAX_VIDEO_FRAMES:
                    ok, frame = capture.read()

                    if not ok:
                        break

                    current_frame = frame_number
                    frame_number += 1

                    if current_frame % frame_step != 0:
                        continue

                    status.write(
                        f"Analyzing frame {current_frame} "
                        f"(sample {sampled_count + 1})..."
                    )

                    frame_results = {
                        name: run_inference(detector, frame)
                        for name, detector in detectors.items()
                    }

                    if first_frame_results is None:
                        first_frame_results = frame_results

                    record = {
                        "frame_number": current_frame,
                        "timestamp_seconds": (
                            round(current_frame / fps, 3)
                            if fps > 0
                            else None
                        ),
                        "results": {
                            name: {
                                "inference_ms": result["inference_ms"],
                                "report": result["report"],
                            }
                            for name, result in frame_results.items()
                        },
                    }

                    if len(frame_results) == 2:
                        record["detection_agreement"] = compare_detections(
                            frame_results["PyTorch"]["detections"],
                            frame_results["ONNX"]["detections"],
                        )

                    frame_records.append(record)
                    sampled_count += 1

                    if total_frames > 0:
                        progress.progress(
                            min(1.0, frame_number / total_frames)
                        )

                capture.release()
                capture = None
                status.empty()

                if not frame_records:
                    st.warning("No readable video frames were sampled.")
                else:
                    st.success(
                        f"Analyzed {sampled_count} sampled frame(s)."
                    )

                    if first_frame_results:
                        st.subheader("First sampled frame")

                        if len(first_frame_results) == 1:
                            name, result = next(
                                iter(first_frame_results.items())
                            )
                            display_result(name, result)
                        else:
                            pt_col, onnx_col = st.columns(2)

                            with pt_col:
                                display_result(
                                    "PyTorch Native",
                                    first_frame_results["PyTorch"],
                                )

                            with onnx_col:
                                display_result(
                                    "ONNX",
                                    first_frame_results["ONNX"],
                                )

                            display_runtime_comparison(first_frame_results)

                    summary_rows = []

                    for record in frame_records:
                        row = {
                            "frame": record["frame_number"],
                            "timestamp_seconds": record["timestamp_seconds"],
                        }

                        for name, result in record["results"].items():
                            row[f"{name}_ms"] = result["inference_ms"]
                            row[f"{name}_status"] = result["report"]["status"]
                            row[f"{name}_helmets"] = result["report"]["helmet_count"]
                            row[f"{name}_heads"] = result["report"]["head_count"]

                        if "detection_agreement" in record:
                            row["agreement_percent"] = record[
                                "detection_agreement"
                            ]["agreement_percent"]

                        summary_rows.append(row)

                    st.subheader("Sampled-frame results")
                    st.dataframe(
                        pd.DataFrame(summary_rows),
                        use_container_width=True,
                        hide_index=True,
                    )

                    video_report = {
                        "application": "VisionGuard AI",
                        "input_type": "video",
                        "video_name": uploaded.name,
                        "sampled_frames": sampled_count,
                        "sampling_interval_frames": frame_step,
                        "model_paths": {
                            name: str(path)
                            for name, path in selected_models.items()
                        },
                        "frame_results": frame_records,
                        "limitations": [
                            "Only sampled frames were analyzed.",
                            "Events between sampled frames may be missed.",
                            "Model agreement does not establish detection accuracy.",
                            "Human review is required for safety-related decisions.",
                        ],
                    }

                    st.download_button(
                        "📥 Download video comparison JSON",
                        data=json.dumps(video_report, indent=2),
                        file_name="visionguard_video_report.json",
                        mime="application/json",
                    )

            except Exception as exc:
                st.error(f"Video inference failed: {exc}")

            finally:
                if capture is not None:
                    capture.release()

                if temporary_path and Path(temporary_path).exists():
                    Path(temporary_path).unlink(missing_ok=True)


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------

st.divider()
st.caption(
    "VisionGuard AI · Experimental computer vision · "
    "OpenCV visual processing · Human-in-the-loop inspection"
)
