# Architecture and experiment record

## Components

```text
Browser (vanilla JS, CSS, HTML)
  ├─ local Node static server :4173
  └─ REST requests → FastAPI :8000
       ├─ lazy Hugging Face CLIP loader
       ├─ uploaded scene scoring with editable CLIP text prompts
       ├─ PyTorch FGSM / PGD / shared UAP / patch attack paths
       ├─ FGSM / PGD perturbation-strength sweep using one source image
       ├─ clean / adversarial prompt-sensitivity comparison
       ├─ disclosed heuristic defense endpoint
       └─ SQLite experiment history + CSV export
```

The frontend has no external JavaScript package dependency; it serves locally and only needs the API for actual inference. The model is loaded on the first attack request. `eval()` mode is used and attack gradients are computed with respect to image pixels only. No optimizer updates the model weights.

## Image objective

For RGB image pixels `x ∈ [0,1]^(3×224×224)`, the backend normalizes channels using CLIP's published mean/std, encodes the image and text, and L2-normalizes both embeddings. The objective is the cosine similarity `s(x,t) = f_image(x) · f_text(t)` for the selected target text. The reference prompt score is also measured before and after as a comparison.

For FGSM, the backend computes `g = ∇x s(x,t)` and performs `x' = clip_[0,1](x + ε sign(g))`, followed by projection to `||x'−x||∞ ≤ ε` (with the projection step applied in all methods). PGD repeats this using configurable step size `α`; optional random initialization samples inside the epsilon ball. Inputs are resized to 224×224, so the reported perturbation norm is measured on that attack tensor.

An increased target similarity is an observed objective response. It does not alone establish retrieval failure or semantic validity. This build does not calculate retrieval accuracy or a validated attack success rate.

## Defense heuristic

`POST /api/defense/analyze` computes a visible rule-based score from similarity drop, measured L∞ distance, and caller-supplied transformation variance. It returns its component features, fixed threshold, predicted routing label, and recommendation. This score is not a trained detector, has no calibration study, and has no precision/recall/F1 result. The UI shows those evaluation metrics as unavailable.

## Persistent record

Each completed attack stores a UUID, UTC timestamp, model identifier, method, epsilon (zero for patches), iterations, mean reference similarity before/after, elapsed time, and a JSON detail object in SQLite. The detail object retains attack/prompt configuration, source image filename, and applicable measured values. Prompt comparisons are saved with prompt scores and highest-scoring prompt names. Uploaded image-ranking requests are not saved. Images are returned for immediate viewing and are not stored in the database. CSV export includes the JSON detail column.

## Research metrics and scope

The attack laboratory reports actual pixel-space L-infinity magnitude and attack-generation time where applicable. It plots clean and adversarial CLIP reference cosine similarity across selected strengths. These scores measure image-text alignment; they are not class predictions or calibrated probabilities. Clean accuracy, robust accuracy, attack success rate, and accuracy drop are marked unavailable without a labeled classification evaluator. When classification labels exist, attack success rate must divide only by examples correctly classified before the attack.

Prompt comparison reports cosine scores for editable standard, descriptive, and verification-oriented prompts on both the clean and adversarial image. Highest-score indicators describe per-image comparisons and are not defense claims.

## Simulation boundary

Camera and road pages accept a local image and use CLIP to score editable descriptions. They do not read a camera stream, identify people, verify credentials, detect objects, recommend access, or control a vehicle. Relative scores are not calibrated probabilities. Camera results require human verification; road results must not guide real driving.

## Validation status

The browser JavaScript, Node static server, package JSON, and startup PowerShell script were syntax-checked. Backend unit tests cover gradient bounds, shared UAP behavior, patch placement, and model-parameter preservation. API smoke tests cover health and experiment listing. A real CPU CLIP FGSM request completed against a generated synthetic image; documented measurements are a smoke run, not a benchmark or success claim. Dataset-backed classification evaluation, text-preserving and joint attacks, gradient conflict extraction, multi-model benchmarks, detector validation, and WebSocket progress remain future work.
