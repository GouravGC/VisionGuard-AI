# 🛡️ VisionGuard AI

### OpenCV 5-Powered PPE Visual Inspection with Human-in-the-Loop Review

[![CI/CD](https://github.com/GouravGC/VisionGuard-AI/actions/workflows/ci.yml/badge.svg)](https://github.com/GouravGC/VisionGuard-AI/actions)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Render-46E3B7?logo=render&logoColor=white)](https://visionguard-ai-latest.onrender.com/)
[![Docker Image](https://img.shields.io/badge/Container-GHCR-2496ED?logo=docker&logoColor=white)](https://github.com/GouravGC/VisionGuard-AI/pkgs/container/visionguard-ai)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.x-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)

**VisionGuard AI** is a computer vision prototype for inspecting workplace images and sampled video frames for helmet- and head-related visual evidence. It combines OpenCV image/video processing with a trained YOLO object detection model and provides annotated outputs, structured reports, and explicit human-review indicators.

The goal is to make visual inspection easier while keeping uncertainty visible. A model prediction is not treated as a confirmed safety violation.

## 🚀 Live Links

| Resource | Link |
|---|---|
| Live application | [Open VisionGuard AI](https://visionguard-ai-latest.onrender.com/) |
| Source code | [GitHub repository](https://github.com/GouravGC/VisionGuard-AI) |
| CI/CD workflow | [GitHub Actions](https://github.com/GouravGC/VisionGuard-AI/actions) |
| Container package | [GitHub Container Registry](https://github.com/GouravGC/VisionGuard-AI/pkgs/container/visionguard-ai) |

The application is deployed on Render using a Docker image published to GitHub Container Registry. The deployed instance uses CPU inference; GPU acceleration is not assumed.

## ✨ Features

- **Image inspection:** Analyze uploaded workplace images for helmet- and head-like regions.
- **Video inspection:** Process sampled video frames rather than running inference on every frame.
- **Visual annotations:** Display bounding boxes and associated detection confidence.
- **Human-in-the-loop review:** Flag potentially concerning observations for manual verification.
- **Structured output:** Export inspection results as JSON and download annotated images.
- **Inference comparison:** Compare PyTorch and ONNX execution paths where supported by the application.
- **Automated testing:** Use pytest to validate core analysis behavior.
- **Containerized deployment:** Package the application and model artifacts in Docker.
- **CI/CD:** Use GitHub Actions for testing, image publishing, and the configured Render deployment trigger.

## 🧠 How It Works

```text
Workplace Image or Video
          |
          v
OpenCV Image / Video Processing
          |
          v
YOLO Object Detection
          |
          v
Helmet / Head Detection Filtering
          |
          v
Evidence Analysis and Annotation
          |
          v
Human-Review Status + JSON Report
```

### Processing pipeline

1. **Input:** The user uploads an image or supported video.
2. **Preprocessing:** OpenCV handles image decoding and video-frame extraction.
3. **Inference:** The trained YOLO model produces candidate detections.
4. **Filtering:** The application focuses on the helmet and head classes.
5. **Analysis:** The application assigns an evidence status based on the detections.
6. **Output:** Annotated visual evidence and structured results are presented for review.

The application is designed for inspection assistance, not autonomous enforcement.

## 🏗️ Technology Stack

| Component | Technology |
|---|---|
| Language | Python 3.11 |
| UI | Streamlit |
| Computer vision | OpenCV 5.x |
| Object detection | Ultralytics YOLO |
| Model formats | PyTorch `.pt` and ONNX `.onnx` |
| ONNX inference | ONNX Runtime |
| Data handling | NumPy, Pandas |
| Testing | pytest |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Container registry | GitHub Container Registry |
| Application hosting | Render |

## 📊 Model and Evaluation

The project uses a YOLOv8n model trained on the [Hard Hat Detection dataset](https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection).

The training workflow converted Pascal VOC annotations into YOLO format and used separate training and validation splits.

### Validation results

| Class | Precision | Recall | mAP@50 | mAP@50–95 |
|---|---:|---:|---:|---:|
| Helmet | 0.928 | 0.855 | 0.928 | 0.597 |
| Head | 0.901 | 0.838 | 0.904 | 0.581 |
| Person | 1.000 | 0.000 | 0.024 | 0.013 |
| Overall | 0.943 | 0.564 | 0.619 | 0.397 |

These figures are from the project's recorded validation evaluation. They are not a guarantee of performance on new sites, camera angles, lighting conditions, or video footage.

**Important:** Person-class performance is poor. The application excludes that class from its analysis rather than presenting it as reliable. Helmet and head detections are visual indicators and do not establish that a particular worker is compliant or non-compliant.

## ⚙️ Local Setup

### Requirements

- Python 3.11
- Git
- Sufficient RAM for the model and its inference dependencies
- The repository's trained model artifacts

### 1. Clone the repository

```bash
git clone https://github.com/GouravGC/VisionGuard-AI.git
cd VisionGuard-AI
```

### 2. Create a virtual environment

Using Conda:

```bash
conda create -n visionguard python=3.11.10 -y
conda activate visionguard
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Verify the installation

```bash
python -m pytest -q
python -m pip check
```

### 5. Launch the application

```bash
streamlit run app.py
```

Streamlit will display the local URL in the terminal.

### Model artifacts

The repository includes these model files:

- `artifacts/models/visionguard_helmet.pt`
- `artifacts/models/visionguard_helmet.onnx`

The default model path is configured by the application. To select a different PyTorch model path, set `VISIONGUARD_MODEL_PATH` to the desired file.

## 🐳 Docker Deployment

The repository includes a Dockerfile and Render configuration.

The intended container image is:

```text
ghcr.io/gouravgc/visionguard-ai:latest
```

A commit-specific image tag is also published by the GitHub Actions workflow.

To pull the image manually:

```bash
docker pull ghcr.io/gouravgc/visionguard-ai:latest
```

The image must be publicly accessible or the user must authenticate to GHCR before pulling it. The command requires Docker to be installed locally.

## 🔄 CI/CD Workflow

The intended pipeline is:

1. A push or pull request triggers automated checks.
2. GitHub Actions installs dependencies and runs pytest.
3. The workflow checks dependency consistency.
4. On a push to `main`, Docker builds the application image.
5. The image is published to GitHub Container Registry.
6. The configured Render deploy hook triggers a deployment after image publishing succeeds.

Check the [Actions page](https://github.com/GouravGC/VisionGuard-AI/actions) for the actual status of each run. A successful test job alone does not prove that publishing and deployment also succeeded.

## ☁️ AWS Integration Status

**Status: Not yet implemented.**

AWS integration is a planned extension and is not represented as a deployed capability in the current application.

A potential next step is Amazon S3 storage for annotated inspection images and JSON reports, using restricted IAM permissions. This would provide persistent evidence storage while keeping model inference on the existing application host.

Any AWS services will be documented as implemented only after they have been configured, tested, and connected to the application.

## 🔐 Responsible Use and Limitations

VisionGuard AI is a research prototype and is not a certified workplace safety system.

- A missing helmet detection does not prove that a worker is violating a safety policy.
- Head and helmet detections are not linked reliably to individual workers.
- The person class has poor validation performance and is excluded from the current analysis.
- Video analysis samples frames and can miss events between samples.
- Inference latency varies with hardware, image resolution, model initialization, and runtime.
- CPU-only hosting does not provide GPU acceleration.
- Predictions must be reviewed by a qualified human before any operational decision.
- Avoid uploading confidential footage or personally identifiable information unless you are authorized to do so.

## 📁 Repository Structure

```text
VisionGuard-AI/
├── .github/
│   └── workflows/
│       └── ci.yml
├── artifacts/
│   ├── models/
│   ├── evaluation/
│   ├── training/
│   └── config/
├── docs/
│   ├── architecture.md
│   └── model-card.md
├── src/
│   └── visionguard/
├── tests/
├── app.py
├── Dockerfile
├── .dockerignore
├── pytest.ini
├── render.yaml
├── requirements.txt
└── README.md
```

## 🗺️ Roadmap

- [x] Train a YOLO-based helmet/head detector.
- [x] Export PyTorch and ONNX model artifacts.
- [x] Implement image and sampled-video inspection.
- [x] Provide annotated outputs and structured reports.
- [x] Add automated tests and container configuration.
- [x] Configure GitHub Actions and Render deployment workflow.
- [ ] Complete and publish final demo and evaluation evidence.
- [ ] Implement and verify AWS-backed evidence storage.
- [ ] Expand evaluation on varied lighting, occlusion, and camera conditions.

## 📚 Documentation

- [System architecture](docs/architecture.md)
- [Model card](docs/model-card.md)
- [GitHub Actions](https://github.com/GouravGC/VisionGuard-AI/actions)
- [Live application](https://visionguard-ai-latest.onrender.com/)

## 📄 License

Add or confirm the repository's chosen license before distributing this project. Also review the original dataset and model-library terms before reuse.

---

**Built as a computer vision prototype for PPE inspection, with an emphasis on transparent uncertainty, reproducible evaluation, and human review.**
