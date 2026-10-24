
# VisionGuard AI

**OpenCV 5-powered visual PPE evidence inspector with human-in-the-loop review.**

VisionGuard AI analyzes workplace images and sampled video frames to identify helmet/head-like regions, annotate visual evidence, and flag frames that may need additional review.

## Features

- OpenCV image and video processing
- YOLO-based helmet/head detection
- Annotated images and downloadable JSON reports
- Sampled video inspection
- Human-review flags and explicit uncertainty
- Pytest, GitHub Actions CI, Docker, and Render configuration

## Quick start

Requires Python 3.11 and the dependencies in `requirements.txt`.

```bash
python -m pip install -r requirements.txt
python -m pytest -q
```

Place trained model weights at:

`artifacts/models/visionguard_helmet.pt`

Alternatively, set `VISIONGUARD_MODEL_PATH` to the model's path.

Launch the dashboard:

```bash
streamlit run app.py
```

## Model limitations

The person class had poor validation performance and is excluded from the app's analysis. Head and helmet detections are scene-level indicators and do not prove worker-level PPE compliance. Video analysis samples frames and may miss events between samples.

This project is a research prototype, not a certified safety system. Human review is required.

## Architecture

See [`docs/architecture.md`](docs/architecture.md) and [`docs/model-card.md`](docs/model-card.md).

## Deployment

The repository includes a Dockerfile, Render blueprint, and GitHub Actions workflow. Configure the model artifact path and ensure the weights are available to the deployed service before enabling public inference.

## Competition evidence

Before submission, document the OpenCV 5 processing actually used, test results, known failure cases, architecture, and the meaningful role of any AWS services that are implemented.
