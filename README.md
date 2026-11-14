# Sentinel AI

An academic research interface for controlled image–text adversarial experiments. The working model path uses the open-weight Hugging Face CLIP ViT-B/32 model and PyTorch autograd. It generates measured image-text similarity scores and saves completed runs to a local SQLite database.

## Current implementation

| Area | Status |
|---|---|
| Responsive research interface, dark/light theme | Implemented; theme preference persists in the browser |
| CLIP baseline and image input | Implemented through the attack workflow |
| FGSM / PGD / universal perturbation / patch | Implemented with measured CLIP objectives and local experiment records |
| Experiment history and CSV export | Implemented; records are local in `data/experiments.sqlite3` |
| Adaptive FGSM / PGD strength sweep | Implemented; uses the same uploaded image, model and attack settings at 2–8 measured ε values |
| Prompt sensitivity comparison | Implemented; compares three editable prompts against the same clean and adversarial images using actual CLIP cosine similarity |
| Robustness metrics dashboard | Implemented; reports measured L∞ and execution time, and marks classification metrics unavailable without labeled predictions |
| Anomaly score | Transparent heuristic; research-only, not a trained or validated detector |
| Camera / road image tests | Upload local images and compare editable scene descriptions using actual CLIP scores |
| Textual / joint attacks, gradient conflict, second model, trained detector | Not implemented in this build |

Similarity change does not automatically mean an attack succeeded. The attack objective increases similarity to a chosen target prompt; both target and reference scores are returned for inspection. CLIP similarity-only runs cannot provide clean accuracy, robust accuracy, attack success rate, or accuracy drop without labeled classification predictions. The dashboard defines these metrics and leaves them unavailable instead of estimating them from similarity.

## Requirements

- Windows 10/11, Python 3.10 or newer, and Node.js 18 or newer.
- Internet access for the first dependency install and first CLIP model download.
- CPU works for small experiments. CUDA is used automatically when the installed PyTorch build detects it.
- Disk space for Python/PyTorch packages and the model files (downloaded to `data/models`).

## Start

Open PowerShell in this project folder and run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\start.ps1
```

Then open [http://127.0.0.1:4173](http://127.0.0.1:4173). The UI server runs in that terminal; the API starts in a hidden window on port 8000. Close the terminal to stop the UI. To stop the API, find and close the Python process running `uvicorn`.

The starter creates a local virtual environment, installs `backend/requirements.txt`, then starts both services. If you want to choose the PyTorch wheel explicitly (for example, CPU-only), install PyTorch using the selector on [pytorch.org](https://pytorch.org/get-started/locally/) first, then install the remaining requirements.

## Manual startup

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:SENTINEL_DB = "..\data\experiments.sqlite3"
$env:HF_HOME = "..\data\models"
python -m uvicorn sentinel.main:app --host 127.0.0.1 --port 8000
```

In a second terminal:

```powershell
cd frontend
node serve.mjs
```

The model `openai/clip-vit-base-patch32` is downloaded on the first attack run. Configure `SENTINEL_MODEL_ID`, `SENTINEL_DEVICE`, and `SENTINEL_DB` with environment variables before launching the API if needed. The backend reads `SENTINEL_DEVICE=auto` by default and uses CUDA when available.

## Attack method notes

- **FGSM:** Computes the gradient of target-prompt cosine similarity with respect to the image, takes one sign-gradient step, and clips the result to both the valid pixel interval [0, 1] and the configured L∞ ball around the source.
- **PGD:** Repeats sign-gradient ascent with a configurable step size. Optional random initialization begins inside the L∞ ball. Projection and pixel clipping happen at every step.
- **Universal perturbation:** Optimizes one shared RGB delta against the mean target-prompt cosine across a batch of 1–8 images. The shared delta is projected into the configured L∞ budget each step.
- **Adversarial patch:** Optimizes one localized RGB patch with Adam against mean target-prompt cosine across 1–8 images. Patch area and position are configurable; an L∞ bound is not used for this localized attack.
- **Camera / road uploads:** CLIP ranks 2–8 user-editable scene descriptions against an uploaded image. The within-list softmax is relative to those descriptions and is not a calibrated probability, identity check, object detector, or safety signal.
- Image normalization uses the CLIP channel mean and standard deviation. Optimization is performed in RGB pixel space before normalization. Model parameters are not updated.
- The backend reports clean and adversarial similarity for a reference caption and target caption, actual L∞ distance, elapsed time, device, and output images. A successful run is not guaranteed.

## API

- `GET /api/health` — runtime/model status and available device.
- `POST /api/attack` — multipart image, reference/target text, FGSM or PGD settings; executes and stores a measured run.
- `POST /api/attack/sweep` — multipart image, prompts, FGSM/PGD method, comma-separated strengths, and fixed optimization settings; returns measured scores and adversarial images for each strength and saves each run.
- `POST /api/experiments/prompt-comparison` — multipart clean/adversarial image pair and three editable prompts; returns actual cosine similarities and stores the comparison in experiment details.
- `POST /api/attack/universal` — batch images and shared-perturbation settings; returns per-image scores.
- `POST /api/attack/patch` — batch images, patch size/position and optimization iterations; returns optimized patch and per-image scores.
- `POST /api/analyze-image` — image plus 2–8 candidate scene descriptions; returns actual CLIP cosine scores and relative rankings.
- `GET /api/experiments` — saved results.
- `GET /api/experiments.csv` — CSV export.
- `GET /api/summary` — aggregate measured similarities, or an empty result when no experiments exist.
- `POST /api/defense/analyze` — explicit heuristic anomaly score and conservative recommendation.
- `GET /docs` — interactive API documentation while the API is running.

## Checks

With the backend environment installed, run the backend unit and API smoke tests from `backend`:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Run this from the `backend` folder after the first setup.

The attack tests use small CPU tensors and do not download the CLIP model. The API tests check runtime metadata and experiment-list behavior. The local verification run below records the checks that were completed.

### Local verification run

Python 3.12, CPU PyTorch, and the backend dependencies were installed and the checks were run. Result: **7 tests passed** (the installed Starlette version emits one `httpx` deprecation warning). The CLIP weights also downloaded and actual CPU CLIP requests exercised scene ranking, FGSM, shared UAP, and patch optimization on generated images. These are execution smoke checks, not dataset benchmarks or claims of attack success:

| Measurement | Value |
|---|---:|
| Model | `openai/clip-vit-base-patch32` |
| Method / budget | FGSM / ε = 0.03 |
| Reference cosine, clean → adversarial | 0.29936 → 0.32150 |
| Target cosine, clean → adversarial | 0.24044 → 0.25286 |
| Measured L∞ distance | 0.03000 |

The reference score increased in this run; it should not be described as a successful attack. The run was recorded in a separate verification database, not in the app's experiment history.

## Limitations and safe use

Camera and road modules analyze uploaded images with CLIP text similarity. They do not identify people, verify credentials, detect objects, or control vehicles. The adaptive detector is a disclosed heuristic with no held-out evaluation. Do not use these scores or code to make real access-control or driving decisions. Use images you are authorized to analyze.

This build does not claim the complete research agenda in the original brief is finished. In particular, text-preserving attacks, joint multimodal optimization, gradient conflict extraction, multi-model benchmarks, detector training/validation, and WebSocket progress remain future work.
