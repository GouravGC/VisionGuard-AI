# VisionGuard AI — Model Handoff

## Included
- `models/visionguard_helmet.pt`: native Ultralytics/PyTorch model
- `models/visionguard_helmet.onnx`: ONNX export
- `training/`: training checkpoints, arguments and results
- `evaluation/`: validation plots and CSV metrics, when generated
- `config/data.yaml`: dataset configuration
- `model_metadata.json`: model and training metadata

## Classes
The training configuration uses helmet, head, and person.
The person class performed poorly in the recorded validation run.
Do not use this model as a certified safety system or as the sole basis
for real-world safety decisions.

## Reproducibility
The original dataset is not included in this bundle. Obtain it separately
from its source and review its license before redistribution.
Pin and verify Python, PyTorch, Ultralytics, ONNX and OpenCV versions
when building the deployment environment.

## Planned deployment
The application, OpenCV 5 pipeline, human-review flow, AWS storage,
serverless processing, monitoring, CI/CD and deployment configuration
will be added in subsequent project stages. They are not implemented
merely by exporting these model files.
