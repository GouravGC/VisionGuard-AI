
# VisionGuard AI Model Card

## Intended use

Experimental helmet/head visual evidence review in workplace imagery. This model is not certified for occupational safety decisions.

## Training data

Hard Hat Detection dataset, reported as CC0-1.0 by the dataset source. Verify the original dataset terms and attribution requirements before submission.

## Model

YOLOv8 nano initialized from pretrained weights and fine-tuned for 10 epochs on a Tesla T4 GPU. Input size: 416 pixels.

## Validation results

Reported validation metrics after the 10-epoch run:

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| Helmet | 0.928 | 0.855 | 0.928 | 0.597 |
| Head | 0.901 | 0.838 | 0.904 | 0.581 |
| Person | 1.000 | 0.000 | 0.024 | 0.013 |
| Overall | 0.943 | 0.564 | 0.619 | 0.397 |

These results are from the recorded validation run and are not a guarantee of performance on new environments.

## Known limitations

The person class performed poorly and is excluded from the app's analysis. The model may miss objects or produce false detections under unusual lighting, occlusion, camera angles, or domain shift.

The app's rules operate at scene level and cannot confirm whether a specific worker is wearing a helmet correctly.

## Human oversight

All findings require human review. An absence of detections must never be interpreted as proof of a safe scene.
