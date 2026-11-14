# SENTINEL AI implementation audit

## Existing architecture and working features

- **Frontend:** vanilla JavaScript, HTML and CSS, served by a small Node.js static server. Existing navigation, responsive layout, dark/light theme, upload previews, progress indicators and API error messages are implemented.
- **Backend:** FastAPI REST API with PyTorch and Hugging Face CLIP ViT-B/32, loaded lazily and cached in process. CPU is supported; CUDA is selected when available.
- **Implemented attacks:** FGSM and PGD operate on real image gradients in a projected pixel-space L-infinity ball. Shared universal perturbation and localized patch optimization are also implemented with measured CLIP objectives.
- **Inference:** CLIP computes real image and text embeddings/similarities for user-provided images and editable scene descriptions. The interface states that relative scores are not calibrated probabilities.
- **Records:** attack runs are stored in local SQLite and can be listed and exported as CSV. Uploaded scene-ranking requests and their images are not saved.
- **Defense:** the API exposes a transparent rule-based anomaly heuristic. It is not a trained or validated detector; held-out precision, recall and F1 are unavailable.
- **Tests and setup:** backend unit/API tests, a PowerShell starter, dependency list, and architecture/readme documentation exist. A local Python environment and CLIP cache are present.

## Partial, missing, and placeholder areas

- There is no dataset-backed classification evaluation or ground-truth labeling. CLIP similarity alone cannot provide clean accuracy, robust accuracy, attack success rate, or accuracy drop.
- Existing attack forms support a single FGSM/PGD run, but not a controlled sweep over several strengths with one source image and consistent configuration.
- Existing image-text ranking accepts editable prompts for a clean image only; it does not compare prompt behavior between clean and adversarial versions.
- The overview charts are empty placeholders. Model comparison shows that a second model is unavailable. Camera and road pages are local image-ranking interfaces, not camera/object-detection/vehicle systems.
- The experiment table records basic attack metadata and similarities, but does not persist prompt-sensitivity experiments or a full attack-configuration record in a directly comparable form.
- The project folder has no Git repository or GitHub remote. The local database, virtual environment and downloaded model cache are excluded from source control.

## Minimal implementation plan

1. Keep the existing frontend, FastAPI backend, CLIP loader, attack functions, SQLite store, styling and navigation.
2. Add a measured FGSM/PGD strength-sweep API and an interactive attack-lab view that reuses one uploaded source image and the existing model/attack code.
3. Add clean/adversarial prompt comparison using actual CLIP cosine similarities, and record the experiment configuration/results in existing SQLite JSON details.
4. Add research metric definitions and show only applicable measurements: L-infinity and attack time for similarity-only runs; classification metrics stay unavailable without labeled classification predictions.
5. Extend CSV/history as practical and update run documentation. No synthetic scores or fabricated accuracy statistics.
