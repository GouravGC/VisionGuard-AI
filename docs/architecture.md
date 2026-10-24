
# VisionGuard AI Architecture

## Pipeline

1. User uploads an image or video through Streamlit.
2. OpenCV decodes the image or samples video frames.
3. A trained YOLO model identifies helmet and head-like regions.
4. The analysis layer summarizes detections and uncertainty.
5. Frames with possible missing PPE evidence are flagged for human review.
6. Annotated images and JSON reports are presented to the user.

## Agentic vision behavior

Visual results influence the next action: frames with potential concerns are flagged for human review, while inconclusive frames are explicitly reported as insufficient evidence. The video workflow evaluates multiple sampled frames and aggregates review flags.

This is a prototype workflow, not a fully autonomous safety agent. It does not independently dispatch personnel or make final compliance decisions.

## Technology

- OpenCV 5 for image/video processing and visual annotations
- Ultralytics YOLO for object detection
- Streamlit for the user interface
- Pytest and GitHub Actions for automated testing
- Docker and Render for deployment

## Limitations

- Only sampled video frames are analyzed.
- The model's person-class validation performance was poor; that class is excluded from analysis.
- Image-level detections do not associate helmets with individual workers.
- False positives and false negatives are possible.
- Human verification is required.
