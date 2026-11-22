# Rules and Guidelines for SCC & TTC Demo Project

## Tech Stack
- Python 3.10+, PyTorch, open_clip_torch with model "ViT-B-32", pretrained="openai" (must match OpenAI CLIP ViT-B/32 used in the paper).
- Backend: FastAPI. Frontend: Streamlit (calls the FastAPI endpoints).
- Project layout:
  `backend/{main.py, models.py, attacks.py, defenses.py, metrics.py, utils.py}`
  `frontend/app.py`
  `eval/{eval_pgd.py, eval_defense.py}`
  `tests/`
  `samples/`
  `docs/`
  `README.md`

## Correctness Rules (Strict)
1. Keep images in [0,1] pixel space. Apply CLIP mean/std normalization INSIDE a wrapper module so epsilon is in true pixel units. Clamp to [0,1] after every update.
2. Use L-infinity norm everywhere. Eps values are x/255.
3. Text embeddings: prompt "a photo of a {class}", L2-normalized, computed once and cached.
4. Logits = logit_scale * cosine similarity; use the model's own logit_scale.
5. Model in eval mode, requires_grad False on weights, gradients only wrt the image/delta.
6. Seed everything (torch, numpy, random) for reproducibility. Make augmentation noise seedable.
7. Do not invent hyperparameters. If a value is not in the papers, look at docs/TTC_paper_notes.md and clarify.
8. No fine-tuning or training: use frozen pretrained CLIP only.

## Reference Targets (SCC paper Table 1, ε=1/255, PGD-10, ViT-B/32)
- CIFAR-10: Clean ~85.12%, Adversarial CLIP ~0.74%, TTC ~28.75%, SCC ~59.18%
- STL-10: Clean ~96.40%, Adversarial CLIP ~11.00%, TTC ~76.70%, SCC ~90.50%
- 16-dataset average: SCC robust 51.68%, clean 60.21%
