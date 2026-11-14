from __future__ import annotations

import base64
import io
import json
import os
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = Path(os.getenv("SENTINEL_DB", str(ROOT / "data" / "experiments.sqlite3")))
MODEL_ID = os.getenv("SENTINEL_MODEL_ID", "openai/clip-vit-base-patch32")
DEVICE_NAME = os.getenv("SENTINEL_DEVICE", "auto")
DEVICE = "cuda" if DEVICE_NAME == "auto" and torch.cuda.is_available() else ("cpu" if DEVICE_NAME == "auto" else DEVICE_NAME)
MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)

app = FastAPI(title="Sentinel AI Research API", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
model_lock = Lock()
_model = None
_tokenizer = None
_model_error: str | None = None


def db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute("""CREATE TABLE IF NOT EXISTS experiments (
        id TEXT PRIMARY KEY, created_at TEXT NOT NULL, model TEXT NOT NULL,
        attack TEXT NOT NULL, epsilon REAL NOT NULL, iterations INTEGER NOT NULL,
        clean_similarity REAL NOT NULL, adversarial_similarity REAL NOT NULL,
        defended_similarity REAL, elapsed_ms REAL NOT NULL, details TEXT NOT NULL
    )""")
    con.commit()
    return con


def get_model() -> tuple[CLIPModel, CLIPTokenizer]:
    global _model, _tokenizer, _model_error
    from transformers import CLIPModel, CLIPTokenizer
    if _model is None:
        with model_lock:
            if _model is None:
                try:
                    # The pinned public CLIP repo currently publishes pytorch_model.bin, not safetensors.
                    _model = CLIPModel.from_pretrained(MODEL_ID, use_safetensors=False).to(DEVICE).eval()
                    _model.requires_grad_(False)
                    _tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID)
                    _model_error = None
                except Exception as exc:
                    _model_error = f"Model load failed: {type(exc).__name__}: {exc}"
                    raise HTTPException(503, _model_error) from exc
    assert _tokenizer is not None
    return _model, _tokenizer


def prepare_image(image: Image.Image) -> torch.Tensor:
    image = image.convert("RGB").resize((224, 224), Image.Resampling.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(DEVICE)


async def decode_upload(upload: UploadFile) -> tuple[Image.Image, torch.Tensor]:
    raw = await upload.read()
    if len(raw) > 25 * 1024 * 1024:
        raise HTTPException(413, "Image file exceeds 25 MB.")
    try:
        image = Image.open(io.BytesIO(raw))
        if image.width * image.height > 20_000_000:
            raise HTTPException(413, "Image is too large (maximum 20 megapixels).")
        image.load()
        image = image.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(415, "Upload a valid PNG, JPEG or WEBP image.") from exc
    return image, prepare_image(image)


def parse_prompts(value: str) -> list[str]:
    prompts = [line.strip() for line in value.splitlines() if line.strip()]
    if not 2 <= len(prompts) <= 8:
        raise HTTPException(422, "Provide between 2 and 8 candidate descriptions, one per line.")
    if any(len(prompt) > 250 for prompt in prompts):
        raise HTTPException(422, "Each candidate description must be 250 characters or fewer.")
    return prompts


def save_experiment(attack_name: str, epsilon: float, iterations: int,
                    clean_similarity: float, adversarial_similarity: float,
                    elapsed_ms: float, details: dict) -> str:
    experiment_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()
    with db() as con:
        con.execute("INSERT INTO experiments VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (experiment_id, created_at, MODEL_ID, attack_name, epsilon, iterations,
                     clean_similarity, adversarial_similarity, None, elapsed_ms, json.dumps(details)))
    return experiment_id


def normalized(pixels: torch.Tensor) -> torch.Tensor:
    return (pixels - MEAN.to(pixels.device)) / STD.to(pixels.device)


def image_embedding(model: CLIPModel, pixels: torch.Tensor) -> torch.Tensor:
    emb = model.get_image_features(pixel_values=normalized(pixels))
    return F.normalize(emb, dim=-1)


def gradient_attack(objective, source: torch.Tensor, method: str, epsilon: float,
                    step_size: float, iterations: int, random_start: bool, seed: int = 17):
    """Maximize a differentiable scalar objective in a pixel-space L-infinity ball."""
    torch.manual_seed(seed)
    adversarial = source.detach().clone()
    if method == "PGD" and random_start:
        adversarial = (adversarial + torch.empty_like(adversarial).uniform_(-epsilon, epsilon)).clamp(0, 1).detach()
    count = 1 if method == "FGSM" else iterations
    for _ in range(count):
        adversarial.requires_grad_(True)
        score = objective(adversarial)
        gradient = torch.autograd.grad(score, adversarial, only_inputs=True)[0]
        with torch.no_grad():
            alpha = epsilon if method == "FGSM" else step_size
            adversarial = adversarial + alpha * gradient.sign()
            adversarial = torch.maximum(torch.minimum(adversarial, source + epsilon), source - epsilon).clamp(0, 1).detach()
    return adversarial, count


def universal_perturbation(objective, images: torch.Tensor, epsilon: float,
                           step_size: float, iterations: int, random_start: bool,
                           seed: int = 17) -> tuple[torch.Tensor, torch.Tensor]:
    """Optimize one shared RGB perturbation over a batch of images."""
    torch.manual_seed(seed)
    delta = torch.empty_like(images[:1]).uniform_(-epsilon, epsilon) if random_start else torch.zeros_like(images[:1])
    for _ in range(iterations):
        delta = delta.detach().requires_grad_(True)
        batch = (images + delta).clamp(0, 1)
        gradient = torch.autograd.grad(objective(batch), delta, only_inputs=True)[0]
        with torch.no_grad():
            delta = (delta + step_size * gradient.sign()).clamp(-epsilon, epsilon).detach()
    return (images + delta).clamp(0, 1).detach(), delta.detach()


def overlay_patch(images: torch.Tensor, patch: torch.Tensor,
                  x_center: float, y_center: float) -> tuple[torch.Tensor, torch.Tensor]:
    _, _, height, width = images.shape
    _, _, ph, pw = patch.shape
    left = min(max(round(x_center * width - pw / 2), 0), width - pw)
    top = min(max(round(y_center * height - ph / 2), 0), height - ph)
    mask = torch.zeros((1, 1, height, width), device=images.device)
    mask[:, :, top:top + ph, left:left + pw] = 1
    patch_canvas = F.pad(patch, (left, width - pw - left, top, height - ph - top))
    return images * (1 - mask) + patch_canvas.expand(images.shape[0], -1, -1, -1), mask


def text_embedding(model: CLIPModel, tokenizer: CLIPTokenizer, text: str) -> torch.Tensor:
    tokens = tokenizer([text], padding=True, truncation=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        return F.normalize(model.get_text_features(**tokens), dim=-1)


def to_png_b64(pixels: torch.Tensor) -> str:
    arr = (pixels.detach().clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
    out = io.BytesIO()
    Image.fromarray(arr).save(out, format="PNG")
    return base64.b64encode(out.getvalue()).decode("ascii")


class DefendRequest(BaseModel):
    clean_similarity: float = Field(ge=-1, le=1)
    adversarial_similarity: float = Field(ge=-1, le=1)
    image_linf: float = Field(ge=0)
    transform_std: float = Field(ge=0)


@app.get("/api/health")
def health():
    try:
        import psutil  # optional, do not make it a runtime requirement
        ram_gb = round(psutil.virtual_memory().total / (1024**3), 1)
    except Exception:
        ram_gb = None
    return {"status": "ready", "model": MODEL_ID, "model_loaded": _model is not None,
            "model_error": _model_error, "device": DEVICE,
            "cuda_available": torch.cuda.is_available(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "ram_gb": ram_gb}


@app.post("/api/analyze-image")
async def analyze_image(image: UploadFile = File(...), prompts: str = Form(...)):
    """Score user-provided scene descriptions against an uploaded image with CLIP."""
    candidates = parse_prompts(prompts)
    _, pixels = await decode_upload(image)
    model, tokenizer = get_model()
    texts = tokenizer(candidates, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        image_features = image_embedding(model, pixels)
        text_features = F.normalize(model.get_text_features(**texts), dim=-1)
        scores = (image_features @ text_features.T).squeeze(0)
        probabilities = F.softmax(scores * model.logit_scale.exp(), dim=0)
    ranking = sorted(
        [{"description": prompt, "cosine_similarity": float(score), "relative_score": float(prob)}
         for prompt, score, prob in zip(candidates, scores.cpu(), probabilities.cpu())],
        key=lambda row: row["cosine_similarity"], reverse=True,
    )
    return {"model": MODEL_ID, "device": DEVICE, "ranking": ranking,
            "top_description": ranking[0]["description"], "top_cosine_similarity": ranking[0]["cosine_similarity"],
            "score_note": "Relative softmax over the supplied descriptions; not a calibrated probability or object detector.",
            "safety_note": "Image-text similarity does not verify identity, credentials, road safety, or authorization.",
            "image_png": to_png_b64(pixels)}


@app.get("/api/experiments")
def experiments(limit: int = 100):
    with db() as con:
        rows = con.execute("SELECT * FROM experiments ORDER BY created_at DESC LIMIT ?", (min(max(limit, 1), 500),)).fetchall()
    return [dict(r) | {"details": json.loads(r["details"])} for r in rows]


@app.get("/api/experiments.csv")
def experiments_csv():
    from fastapi.responses import Response
    import csv
    output = io.StringIO()
    fields = ["id", "created_at", "model", "attack", "epsilon", "iterations", "clean_similarity", "adversarial_similarity", "defended_similarity", "elapsed_ms", "details"]
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    with db() as con:
        for row in con.execute("SELECT * FROM experiments ORDER BY created_at DESC"):
            writer.writerow({k: row[k] for k in fields if k != "details"} | {"details": row["details"]})
    return Response(output.getvalue(), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=sentinel-experiments.csv"})


@app.post("/api/attack")
async def attack(image: UploadFile = File(...), reference_text: str = Form(...),
                 target_text: str = Form(...), method: Literal["FGSM", "PGD"] = Form("FGSM"),
                 epsilon: float = Form(0.03), step_size: float = Form(0.005),
                 iterations: int = Form(10), random_start: bool = Form(True), seed: int = Form(17)):
    if not 0 <= epsilon <= 0.25 or not 1 <= iterations <= 100 or not 0 < step_size <= 0.1:
        raise HTTPException(422, "Use epsilon 0–0.25, step size 0–0.1 and 1–100 iterations.")
    if not reference_text.strip() or not target_text.strip():
        raise HTTPException(422, "Both reference and target text are required.")
    _, x = await decode_upload(image)
    model, tokenizer = get_model()
    ref = text_embedding(model, tokenizer, reference_text.strip())
    target = text_embedding(model, tokenizer, target_text.strip())

    def objective(pixels: torch.Tensor) -> torch.Tensor:
        # Increase the target-prompt similarity while the paired reference is provided for comparison.
        return (image_embedding(model, pixels) @ target.T).squeeze()

    start = time.perf_counter()
    with torch.no_grad():
        clean_ref = float((image_embedding(model, x) @ ref.T).item())
        clean_target = float((image_embedding(model, x) @ target.T).item())
    adv, count = gradient_attack(objective, x, method, epsilon, step_size, iterations, random_start, seed)
    elapsed_ms = (time.perf_counter() - start) * 1000
    with torch.no_grad():
        adv_ref = float((image_embedding(model, adv) @ ref.T).item())
        adv_target = float((image_embedding(model, adv) @ target.T).item())
    linf = float((adv - x).abs().max().item())
    result = {"model": MODEL_ID, "device": DEVICE, "attack": method, "epsilon": epsilon,
              "iterations": count, "clean_similarity": clean_ref,
              "adversarial_similarity": adv_ref, "target_similarity_clean": clean_target,
              "target_similarity_adversarial": adv_target, "linf": linf,
              "elapsed_ms": elapsed_ms, "original_png": to_png_b64(x), "adversarial_png": to_png_b64(adv),
              "perturbation_png": to_png_b64(((adv - x) / max(epsilon, 1e-8) * 0.5 + 0.5).clamp(0, 1))}
    details = {k: result[k] for k in ("device", "target_similarity_clean", "target_similarity_adversarial", "linf")}
    details.update({"dataset_or_image": image.filename or "uploaded image",
                    "attack_configuration": {"method": method, "epsilon": epsilon,
                                             "step_size": step_size, "iterations": count,
                                             "random_start": random_start, "seed": seed},
                    "prompt_configuration": {"reference": reference_text.strip(),
                                              "target": target_text.strip()}})
    result["id"] = save_experiment(method, epsilon, count, clean_ref, adv_ref, elapsed_ms, details)
    return result


def parse_sweep_strengths(strengths: str) -> list[float]:
    try:
        epsilons = sorted({float(part.strip()) for part in strengths.split(",") if part.strip()})
    except ValueError as exc:
        raise HTTPException(422, "Enter perturbation strengths as comma-separated numbers.") from exc
    if not 2 <= len(epsilons) <= 8 or any(not 0 < value <= 0.25 for value in epsilons):
        raise HTTPException(422, "Provide 2–8 distinct strengths between 0 and 0.25.")
    return epsilons


@app.post("/api/attack/sweep")
async def attack_sweep(image: UploadFile = File(...), reference_text: str = Form(...),
                       target_text: str = Form(...), method: Literal["FGSM", "PGD"] = Form("FGSM"),
                       strengths: str = Form("0.01,0.03,0.06,0.1"), step_size: float = Form(0.005),
                       iterations: int = Form(10), random_start: bool = Form(False), seed: int = Form(17)):
    """Run the existing gradient attack at several budgets on one identical input."""
    epsilons = parse_sweep_strengths(strengths)
    if not 1 <= iterations <= 100 or not 0 < step_size <= 0.1:
        raise HTTPException(422, "Use step size 0–0.1 and 1–100 iterations.")
    if not reference_text.strip() or not target_text.strip():
        raise HTTPException(422, "Both reference and target text are required.")

    _, source = await decode_upload(image)
    model, tokenizer = get_model()
    reference = text_embedding(model, tokenizer, reference_text.strip())
    target = text_embedding(model, tokenizer, target_text.strip())

    def objective(pixels: torch.Tensor) -> torch.Tensor:
        return (image_embedding(model, pixels) @ target.T).squeeze()

    with torch.no_grad():
        clean_reference = float((image_embedding(model, source) @ reference.T).item())
        clean_target = float((image_embedding(model, source) @ target.T).item())
    results = []
    for epsilon in epsilons:
        started = time.perf_counter()
        adversarial, count = gradient_attack(objective, source, method, epsilon, step_size,
                                             iterations, random_start, seed)
        elapsed_ms = (time.perf_counter() - started) * 1000
        with torch.no_grad():
            adversarial_reference = float((image_embedding(model, adversarial) @ reference.T).item())
            adversarial_target = float((image_embedding(model, adversarial) @ target.T).item())
        linf = float((adversarial - source).abs().max().item())
        config = {"method": method, "epsilon": epsilon, "step_size": step_size,
                  "iterations": count, "random_start": random_start, "seed": seed,
                  "reference_text": reference_text.strip(), "target_text": target_text.strip(),
                  "source_image": image.filename or "uploaded image"}
        details = {"device": DEVICE, "experiment_type": "adaptive_perturbation_sweep",
                   "attack_configuration": config, "clean_target_similarity": clean_target,
                   "adversarial_target_similarity": adversarial_target, "linf": linf}
        experiment_id = save_experiment(f"{method} strength sweep", epsilon, count,
                                        clean_reference, adversarial_reference, elapsed_ms, details)
        results.append({"id": experiment_id, "epsilon": epsilon, "iterations": count,
                        "clean_similarity": clean_reference,
                        "adversarial_similarity": adversarial_reference,
                        "target_similarity_clean": clean_target,
                        "target_similarity_adversarial": adversarial_target,
                        "linf": linf, "elapsed_ms": elapsed_ms,
                        "adversarial_png": to_png_b64(adversarial)})
    return {"model": MODEL_ID, "device": DEVICE, "attack": method,
            "source_image": image.filename or "uploaded image", "original_png": to_png_b64(source),
            "reference_text": reference_text.strip(), "target_text": target_text.strip(),
            "results": results,
            "metric_note": "Scores are CLIP cosine similarities. No labeled class predictions are available, so accuracy and attack success rate are not calculated."}


@app.post("/api/experiments/prompt-comparison")
async def prompt_comparison(image: UploadFile = File(...), adversarial_image: UploadFile = File(...),
                            standard_prompt: str = Form(...), detailed_prompt: str = Form(...),
                            verification_prompt: str = Form(...), attack_method: str = Form("FGSM"),
                            epsilon: float = Form(0)):
    """Compare editable prompt similarity on a clean image and its measured adversarial version."""
    prompts = [standard_prompt.strip(), detailed_prompt.strip(), verification_prompt.strip()]
    if any(not prompt or len(prompt) > 250 for prompt in prompts):
        raise HTTPException(422, "Enter all three prompts (250 characters or fewer each).")
    _, clean_pixels = await decode_upload(image)
    _, adversarial_pixels = await decode_upload(adversarial_image)
    model, tokenizer = get_model()
    inference_started = time.perf_counter()
    tokens = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        text_features = F.normalize(model.get_text_features(**tokens), dim=-1)
        clean_features = image_embedding(model, clean_pixels)
        adversarial_features = image_embedding(model, adversarial_pixels)
        clean_scores = (clean_features @ text_features.T).squeeze(0).cpu().tolist()
        adversarial_scores = (adversarial_features @ text_features.T).squeeze(0).cpu().tolist()
    elapsed_ms = (time.perf_counter() - inference_started) * 1000
    rows = [{"kind": kind, "prompt": prompt, "clean_similarity": float(clean_score),
             "adversarial_similarity": float(adversarial_score),
             "change": float(adversarial_score - clean_score)}
            for kind, prompt, clean_score, adversarial_score in
            zip(("standard", "detailed", "verification-oriented"), prompts, clean_scores, adversarial_scores)]
    best_clean = max(rows, key=lambda row: row["clean_similarity"])["kind"]
    best_adversarial = max(rows, key=lambda row: row["adversarial_similarity"])["kind"]
    clean_mean = float(np.mean(clean_scores))
    adversarial_mean = float(np.mean(adversarial_scores))
    details = {"experiment_type": "prompt_robustness_comparison", "attack_method": attack_method,
               "epsilon": epsilon, "prompt_configuration": rows,
               "best_clean_prompt": best_clean, "best_adversarial_prompt": best_adversarial,
               "metric_note": "Cosine similarity is not classification probability or accuracy."}
    experiment_id = save_experiment("Prompt comparison", epsilon, 0, clean_mean, adversarial_mean, elapsed_ms,
                                    details)
    return {"id": experiment_id, "model": MODEL_ID, "device": DEVICE, "rows": rows,
            "best_clean_prompt": best_clean, "best_adversarial_prompt": best_adversarial,
            "elapsed_ms": elapsed_ms,
            "metric_note": "CLIP cosine similarities measure image-text alignment, not classification probabilities. Prompt variation is an experimental sensitivity analysis, not a proven defense."}


@app.post("/api/attack/universal")
async def universal_attack(images: list[UploadFile] = File(...), reference_text: str = Form(...),
                           target_text: str = Form(...), epsilon: float = Form(0.03),
                           step_size: float = Form(0.005), iterations: int = Form(20),
                           random_start: bool = Form(True), seed: int = Form(17)):
    if not 1 <= len(images) <= 8:
        raise HTTPException(422, "Upload between 1 and 8 images for this compact universal-attack run.")
    if not reference_text.strip() or not target_text.strip():
        raise HTTPException(422, "Both reference and target text are required.")
    if not 0 <= epsilon <= 0.25 or not 0 < step_size <= 0.1 or not 1 <= iterations <= 100:
        raise HTTPException(422, "Use epsilon 0–0.25, step size 0–0.1 and 1–100 iterations.")
    loaded = [await decode_upload(item) for item in images]
    batch = torch.cat([pixels for _, pixels in loaded], dim=0)
    model, tokenizer = get_model()
    reference = text_embedding(model, tokenizer, reference_text.strip())
    target = text_embedding(model, tokenizer, target_text.strip())
    with torch.no_grad():
        clean_ref = (image_embedding(model, batch) @ reference.T).squeeze(-1)
        clean_target = (image_embedding(model, batch) @ target.T).squeeze(-1)

    def objective(pixels):
        return (image_embedding(model, pixels) @ target.T).mean()

    started = time.perf_counter()
    adversarial, delta = universal_perturbation(objective, batch, epsilon, step_size, iterations, random_start, seed)
    elapsed_ms = (time.perf_counter() - started) * 1000
    with torch.no_grad():
        adv_ref = (image_embedding(model, adversarial) @ reference.T).squeeze(-1)
        adv_target = (image_embedding(model, adversarial) @ target.T).squeeze(-1)
    samples = []
    for idx, (original, adversarial_image) in enumerate(zip(batch, adversarial)):
        samples.append({"original_png": to_png_b64(original.unsqueeze(0)),
                        "adversarial_png": to_png_b64(adversarial_image.unsqueeze(0)),
                        "clean_reference_similarity": float(clean_ref[idx]),
                        "adversarial_reference_similarity": float(adv_ref[idx]),
                        "clean_target_similarity": float(clean_target[idx]),
                        "adversarial_target_similarity": float(adv_target[idx]),
                        "targeted_flip": bool(clean_ref[idx] >= clean_target[idx] and adv_target[idx] > adv_ref[idx])})
    linf = float((adversarial - batch).abs().max().item())
    clean_mean, adv_mean = float(clean_ref.mean()), float(adv_ref.mean())
    details = {"device": DEVICE, "batch_size": len(samples), "shared_delta_linf": linf,
               "targeted_flip_rate": sum(s["targeted_flip"] for s in samples) / len(samples),
               "dataset_or_image": [item.filename or "uploaded image" for item in images],
               "attack_configuration": {"method": "Universal PGD", "epsilon": epsilon,
                                        "step_size": step_size, "iterations": iterations,
                                        "random_start": random_start, "seed": seed},
               "prompt_configuration": {"reference": reference_text.strip(), "target": target_text.strip()}}
    experiment_id = save_experiment("Universal PGD", epsilon, iterations, clean_mean, adv_mean, elapsed_ms, details)
    return {"id": experiment_id, "model": MODEL_ID, "device": DEVICE, "attack": "Universal PGD",
            "epsilon": epsilon, "iterations": iterations, "sample_count": len(samples), "samples": samples,
            "clean_similarity": clean_mean, "adversarial_similarity": adv_mean,
            "target_similarity_clean": float(clean_target.mean()), "target_similarity_adversarial": float(adv_target.mean()),
            "linf": linf, "targeted_flip_rate": details["targeted_flip_rate"], "elapsed_ms": elapsed_ms,
            "shared_perturbation_png": to_png_b64((delta / max(epsilon, 1e-8) * 0.5 + 0.5).clamp(0, 1))}


@app.post("/api/attack/patch")
async def patch_attack(images: list[UploadFile] = File(...), reference_text: str = Form(...),
                       target_text: str = Form(...), patch_fraction: float = Form(0.15),
                       x_center: float = Form(0.5), y_center: float = Form(0.5),
                       iterations: int = Form(40), seed: int = Form(17)):
    if not 1 <= len(images) <= 8:
        raise HTTPException(422, "Upload between 1 and 8 images for this compact patch run.")
    if not reference_text.strip() or not target_text.strip():
        raise HTTPException(422, "Both reference and target text are required.")
    if not 0.05 <= patch_fraction <= 0.5 or not 0 <= x_center <= 1 or not 0 <= y_center <= 1 or not 1 <= iterations <= 200:
        raise HTTPException(422, "Patch size must be 0.05–0.5, position 0–1 and iterations 1–200.")
    loaded = [await decode_upload(item) for item in images]
    batch = torch.cat([pixels for _, pixels in loaded], dim=0)
    model, tokenizer = get_model()
    reference = text_embedding(model, tokenizer, reference_text.strip())
    target = text_embedding(model, tokenizer, target_text.strip())
    with torch.no_grad():
        clean_ref = (image_embedding(model, batch) @ reference.T).squeeze(-1)
        clean_target = (image_embedding(model, batch) @ target.T).squeeze(-1)
    patch_size = max(12, round(224 * patch_fraction))
    torch.manual_seed(seed)
    patch = torch.full((1, 3, patch_size, patch_size), 0.5, device=DEVICE, requires_grad=True)
    optimizer = torch.optim.Adam([patch], lr=0.04)
    started = time.perf_counter()
    for _ in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        patched, _ = overlay_patch(batch, patch, x_center, y_center)
        objective = (image_embedding(model, patched) @ target.T).mean()
        (-objective).backward()
        optimizer.step()
        with torch.no_grad():
            patch.clamp_(0, 1)
    elapsed_ms = (time.perf_counter() - started) * 1000
    with torch.no_grad():
        patched, mask = overlay_patch(batch, patch.detach(), x_center, y_center)
        adv_ref = (image_embedding(model, patched) @ reference.T).squeeze(-1)
        adv_target = (image_embedding(model, patched) @ target.T).squeeze(-1)
    samples = []
    for idx, (original, result_image) in enumerate(zip(batch, patched)):
        samples.append({"original_png": to_png_b64(original.unsqueeze(0)), "adversarial_png": to_png_b64(result_image.unsqueeze(0)),
                        "clean_reference_similarity": float(clean_ref[idx]), "adversarial_reference_similarity": float(adv_ref[idx]),
                        "clean_target_similarity": float(clean_target[idx]), "adversarial_target_similarity": float(adv_target[idx]),
                        "patch_area_fraction": float(mask.mean())})
    clean_mean, adv_mean = float(clean_ref.mean()), float(adv_ref.mean())
    details = {"device": DEVICE, "batch_size": len(samples), "patch_fraction": patch_fraction,
               "x_center": x_center, "y_center": y_center, "target_similarity_clean": float(clean_target.mean()),
               "target_similarity_adversarial": float(adv_target.mean()),
               "dataset_or_image": [item.filename or "uploaded image" for item in images],
               "attack_configuration": {"method": "Adversarial patch", "patch_fraction": patch_fraction,
                                        "x_center": x_center, "y_center": y_center,
                                        "iterations": iterations, "seed": seed},
               "prompt_configuration": {"reference": reference_text.strip(), "target": target_text.strip()}}
    experiment_id = save_experiment("Adversarial patch", 0.0, iterations, clean_mean, adv_mean, elapsed_ms, details)
    return {"id": experiment_id, "model": MODEL_ID, "device": DEVICE, "attack": "Adversarial patch",
            "epsilon": 0.0, "iterations": iterations, "sample_count": len(samples), "samples": samples,
            "clean_similarity": clean_mean, "adversarial_similarity": adv_mean,
            "target_similarity_clean": float(clean_target.mean()), "target_similarity_adversarial": float(adv_target.mean()),
            "linf": None, "patch_fraction": patch_fraction, "patch_png": to_png_b64(patch.detach()), "elapsed_ms": elapsed_ms}


@app.post("/api/defense/analyze")
def defense_analyze(req: DefendRequest):
    # Transparent research heuristic: a measured deviation + transform instability flag.
    drop = max(0.0, req.clean_similarity - req.adversarial_similarity)
    score = float(np.clip(0.55 * min(drop / 0.25, 1) + 0.30 * min(req.image_linf / 0.08, 1) + 0.15 * min(req.transform_std / 0.15, 1), 0, 1))
    suspicious = score >= 0.35
    return {"detector": "transparent heuristic v1", "anomaly_score": score,
            "classification": "image-suspicious" if suspicious else "no strong anomaly detected",
            "recommended_action": "Apply image transformations and request independent review" if suspicious else "Continue with cautious inference",
            "threshold": 0.35, "features": {"similarity_drop": drop, "linf": req.image_linf, "transform_std": req.transform_std},
            "claim": "Research heuristic; not a validated adversarial detector."}


@app.get("/api/summary")
def summary():
    with db() as con:
        rows = con.execute("SELECT clean_similarity, adversarial_similarity, elapsed_ms FROM experiments").fetchall()
    if not rows:
        return {"count": 0, "runs": []}
    return {"count": len(rows), "mean_clean_similarity": float(np.mean([r[0] for r in rows])),
            "mean_adversarial_similarity": float(np.mean([r[1] for r in rows])),
            "mean_elapsed_ms": float(np.mean([r[2] for r in rows]))}
