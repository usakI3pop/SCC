"""FastAPI Application for Adversarial Robustness of Vision-Language Models using PGD and Self-Calibrated Consistency (SCC).
Primary research reference: Liu et al., ICML 2026.
"""

from __future__ import annotations

import base64
import io
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import torch
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from scc.clip_model import get_clip_engine, CLIPEngine
from scc.attacks import pgd_attack, compute_semantic_margin
from scc.defense import scc_defense
from scc.evaluator import run_batch_evaluation
from scc.sample_data import get_sample_catalogue, get_sample_image, CIFAR10_CLASSES, ANIMALS_CLASSES, VEHICLES_CLASSES
from scc.schemas import (
    PredictionItem,
    StageResult,
    ClassifyRequest,
    AttackRequest,
    DefendRequest,
    PipelineRequest,
    PipelineResponse,
    BatchEvaluationResponse,
)

app = FastAPI(
    title="Adversarial Robustness of Vision-Language Models using PGD and Self-Calibrated Consistency",
    description="Research demonstration API reproducing the test-time defense framework from Liu et al., ICML 2026.",
    version="1.0.0",
)

# Enable CORS for local and browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = get_clip_engine()


def _decode_image_payload(image_base64: Optional[str], sample_id: Optional[str]) -> Image.Image:
    """Helper to obtain PIL Image from base64 string or sample ID."""
    if image_base64:
        try:
            # Strip data URI header if present
            if "," in image_base64:
                image_base64 = image_base64.split(",", 1)[1]
            raw_bytes = base64.b64decode(image_base64)
            return Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to decode image_base64: {str(e)}")
    elif sample_id:
        return get_sample_image(sample_id)
    else:
        # Default to dog sample
        return get_sample_image("dog")


def _format_predictions(
    classes: List[str],
    logits: List[float],
    probs: List[float],
    top_k: int = 5,
) -> List[PredictionItem]:
    """Formats top-k predictions with rank, class name, probability, and raw logit."""
    k = min(top_k, len(classes))
    # Sort by probability descending
    indexed = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)[:k]
    items = []
    for rank, (idx, p) in enumerate(indexed, start=1):
        items.append(
            PredictionItem(
                label=classes[idx],
                probability=round(float(p), 4),
                logit=round(float(logits[idx]), 3),
                rank=rank,
            )
        )
    return items


@app.get("/api/health")
def health():
    """System health and model configuration status."""
    return {
        "status": "ready",
        "title": "Adversarial Robustness of Vision-Language Models using PGD and Self-Calibrated Consistency",
        "framework": "Self-Calibrated Consistency (SCC) Test-Time Defense",
        "model": "CLIP ViT-B/32",
        "device": str(engine.device),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }


@app.get("/api/samples")
def list_samples():
    """Returns available benchmark samples and preset class taxonomies."""
    return {
        "samples": get_sample_catalogue(),
        "presets": {
            "cifar10": CIFAR10_CLASSES,
            "animals": ANIMALS_CLASSES,
            "vehicles": VEHICLES_CLASSES,
        },
    }


@app.post("/api/classify")
def classify(req: ClassifyRequest):
    """Performs real zero-shot CLIP ViT-B/32 classification on clean image."""
    start_time = time.perf_counter()
    pil_img = _decode_image_payload(req.image_base64, req.sample_id)
    img_tensor = engine.pil_to_tensor(pil_img)

    classes = [c.strip() for c in req.candidate_labels if c.strip()]
    if not classes:
        classes = CIFAR10_CLASSES

    prompts = [f"a photo of a {c}" for c in classes]
    text_features = engine.encode_text(prompts, normalize=True)

    with torch.no_grad():
        img_features = engine.encode_image(img_tensor, normalize=True)
        logits, probs = engine.compute_logits_and_probs(img_features, text_features, use_temperature=True)

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    logits_list = logits.squeeze(0).cpu().numpy().tolist()
    probs_list = probs.squeeze(0).cpu().numpy().tolist()
    pred_idx = int(torch.argmax(logits, dim=-1).item())
    pred_label = classes[pred_idx]
    confidence = float(probs_list[pred_idx])

    gt_idx = classes.index(req.ground_truth) if req.ground_truth in classes else pred_idx
    margin = compute_semantic_margin(img_features, text_features, gt_idx)
    is_correct = (req.ground_truth == pred_label) if req.ground_truth else None

    top5 = _format_predictions(classes, logits_list, probs_list, top_k=5)

    return StageResult(
        stage_name="Clean CLIP Baseline",
        image_base64=engine.tensor_to_base64(img_tensor),
        predicted_label=pred_label,
        confidence=round(confidence, 4),
        is_correct=is_correct,
        semantic_margin=round(margin, 4),
        top5_predictions=top5,
        execution_time_ms=round(elapsed_ms, 1),
    )


@app.post("/api/attack")
def attack(req: AttackRequest):
    """Executes untargeted white-box PGD attack maximizing cross-entropy against ground truth."""
    pil_img = _decode_image_payload(req.image_base64, req.sample_id)
    clean_tensor = engine.pil_to_tensor(pil_img)

    classes = [c.strip() for c in req.candidate_labels if c.strip()]
    if req.ground_truth not in classes:
        classes = [req.ground_truth] + [c for c in classes if c != req.ground_truth]

    gt_idx = classes.index(req.ground_truth)
    prompts = [f"a photo of a {c}" for c in classes]
    text_features = engine.encode_text(prompts, normalize=True)

    attack_res = pgd_attack(
        engine=engine,
        clean_image=clean_tensor,
        text_features=text_features,
        ground_truth_idx=gt_idx,
        epsilon=req.epsilon,
        alpha=req.alpha,
        steps=req.steps,
        random_init=req.random_init,
    )

    adv_tensor = attack_res["adversarial_image"]
    delta_tensor = attack_res["delta"]
    pred_idx = attack_res["pred_idx"]
    pred_label = classes[pred_idx]
    confidence = float(attack_res["probs"][pred_idx])
    is_correct = (pred_idx == gt_idx)

    top5 = _format_predictions(classes, attack_res["logits"], attack_res["probs"], top_k=5)

    return StageResult(
        stage_name="PGD Adversarial Attack",
        image_base64=engine.tensor_to_base64(adv_tensor),
        predicted_label=pred_label,
        confidence=round(confidence, 4),
        is_correct=is_correct,
        semantic_margin=round(attack_res["semantic_margin"], 4),
        top5_predictions=top5,
        execution_time_ms=round(attack_res["elapsed_ms"], 1),
        perturbation_base64=engine.perturbation_to_base64(delta_tensor, amplification=15.0),
        linf_norm=round(attack_res["linf_norm"], 5),
        linf_pixel=round(attack_res["linf_pixel"], 2),
        l2_norm=round(attack_res["l2_norm"], 4),
    )


@app.post("/api/defend")
def defend(req: DefendRequest):
    """Executes Self-Calibrated Consistency (SCC) test-time defense."""
    pil_img = _decode_image_payload(req.image_base64, None)
    adv_tensor = engine.pil_to_tensor(pil_img)

    classes = [c.strip() for c in req.candidate_labels if c.strip()]
    gt_idx = classes.index(req.ground_truth) if req.ground_truth and req.ground_truth in classes else None

    prompts = [f"a photo of a {c}" for c in classes]
    text_features = engine.encode_text(prompts, normalize=True)

    def_res = scc_defense(
        engine=engine,
        adversarial_image=adv_tensor,
        text_features=text_features,
        ground_truth_idx=gt_idx,
        epsilon_d=req.epsilon_d,
        alpha_d=req.alpha_d,
        steps=req.steps,
        warmup_steps=req.warmup_steps,
        num_views=req.num_views,
        lambda_cm=req.lambda_cm,
        temperature=req.temperature,
        noise_sigma=req.noise_sigma,
    )

    def_tensor = def_res["defended_image"]
    delta_def = def_res["delta"]
    pred_idx = def_res["pred_idx"]
    pred_label = classes[pred_idx]
    confidence = float(def_res["probs"][pred_idx])
    is_correct = (pred_idx == gt_idx) if gt_idx is not None else None

    top5 = _format_predictions(classes, def_res["logits"], def_res["probs"], top_k=5)

    return StageResult(
        stage_name="SCC Test-Time Defense",
        image_base64=engine.tensor_to_base64(def_tensor),
        predicted_label=pred_label,
        confidence=round(confidence, 4),
        is_correct=is_correct,
        semantic_margin=round(def_res["semantic_margin"], 4),
        top5_predictions=top5,
        execution_time_ms=round(def_res["elapsed_ms"], 1),
        perturbation_base64=engine.perturbation_to_base64(delta_def, amplification=15.0),
        linf_norm=round(def_res["delta_norm_inf"], 5),
        linf_pixel=round(def_res["delta_pixel_inf"], 2),
        l2_norm=round(def_res["delta_norm_l2"], 4),
    )


@app.post("/api/pipeline", response_model=PipelineResponse)
def full_pipeline(req: PipelineRequest):
    """End-to-end research comparison pipeline: Clean Classification -> PGD Attack -> SCC Defense."""
    total_start = time.perf_counter()

    pil_img = _decode_image_payload(req.image_base64, req.sample_id)
    clean_tensor = engine.pil_to_tensor(pil_img)

    classes = [c.strip() for c in req.candidate_labels if c.strip()]
    if req.ground_truth not in classes:
        classes = [req.ground_truth] + [c for c in classes if c != req.ground_truth]

    gt_idx = classes.index(req.ground_truth)
    prompts = [f"a photo of a {c}" for c in classes]
    text_features = engine.encode_text(prompts, normalize=True)

    # 1. Clean Stage
    t0 = time.perf_counter()
    with torch.no_grad():
        clean_feat = engine.encode_image(clean_tensor, normalize=True)
        clean_logits, clean_probs = engine.compute_logits_and_probs(clean_feat, text_features, use_temperature=True)
    t_clean = (time.perf_counter() - t0) * 1000.0

    clean_logits_list = clean_logits.squeeze(0).cpu().numpy().tolist()
    clean_probs_list = clean_probs.squeeze(0).cpu().numpy().tolist()
    clean_pred_idx = int(torch.argmax(clean_logits, dim=-1).item())
    clean_pred_label = classes[clean_pred_idx]
    clean_correct = (clean_pred_idx == gt_idx)
    clean_margin = compute_semantic_margin(clean_feat, text_features, gt_idx)

    clean_stage = StageResult(
        stage_name="Clean CLIP Baseline",
        image_base64=engine.tensor_to_base64(clean_tensor),
        predicted_label=clean_pred_label,
        confidence=round(float(clean_probs_list[clean_pred_idx]), 4),
        is_correct=clean_correct,
        semantic_margin=round(clean_margin, 4),
        top5_predictions=_format_predictions(classes, clean_logits_list, clean_probs_list, top_k=5),
        execution_time_ms=round(t_clean, 1),
    )

    # 2. Attack Stage
    attack_res = pgd_attack(
        engine=engine,
        clean_image=clean_tensor,
        text_features=text_features,
        ground_truth_idx=gt_idx,
        epsilon=req.epsilon_a,
        steps=req.steps_a,
        random_init=req.random_init,
    )
    adv_tensor = attack_res["adversarial_image"]
    delta_tensor = attack_res["delta"]
    adv_pred_idx = attack_res["pred_idx"]
    adv_pred_label = classes[adv_pred_idx]
    adv_correct = (adv_pred_idx == gt_idx)
    attack_success = clean_correct and (not adv_correct)

    attack_stage = StageResult(
        stage_name="PGD Adversarial Attack",
        image_base64=engine.tensor_to_base64(adv_tensor),
        predicted_label=adv_pred_label,
        confidence=round(float(attack_res["probs"][adv_pred_idx]), 4),
        is_correct=adv_correct,
        semantic_margin=round(attack_res["semantic_margin"], 4),
        top5_predictions=_format_predictions(classes, attack_res["logits"], attack_res["probs"], top_k=5),
        execution_time_ms=round(attack_res["elapsed_ms"], 1),
        perturbation_base64=engine.perturbation_to_base64(delta_tensor, amplification=15.0),
        linf_norm=round(attack_res["linf_norm"], 5),
        linf_pixel=round(attack_res["linf_pixel"], 2),
        l2_norm=round(attack_res["l2_norm"], 4),
    )

    # 3. Defense Stage (SCC)
    def_res = scc_defense(
        engine=engine,
        adversarial_image=adv_tensor,
        text_features=text_features,
        ground_truth_idx=gt_idx,
        epsilon_d=req.epsilon_d,
        steps=req.steps_d,
        warmup_steps=req.warmup_steps,
        num_views=req.num_views,
        lambda_cm=req.lambda_cm,
        temperature=req.temperature,
        noise_sigma=req.noise_sigma,
    )
    def_tensor = def_res["defended_image"]
    delta_def = def_res["delta"]
    def_pred_idx = def_res["pred_idx"]
    def_pred_label = classes[def_pred_idx]
    def_correct = (def_pred_idx == gt_idx)
    defense_recovered = attack_success and def_correct

    defended_stage = StageResult(
        stage_name="SCC Test-Time Defense",
        image_base64=engine.tensor_to_base64(def_tensor),
        predicted_label=def_pred_label,
        confidence=round(float(def_res["probs"][def_pred_idx]), 4),
        is_correct=def_correct,
        semantic_margin=round(def_res["semantic_margin"], 4),
        top5_predictions=_format_predictions(classes, def_res["logits"], def_res["probs"], top_k=5),
        execution_time_ms=round(def_res["elapsed_ms"], 1),
        perturbation_base64=engine.perturbation_to_base64(delta_def, amplification=15.0),
        linf_norm=round(def_res["delta_norm_inf"], 5),
        linf_pixel=round(def_res["delta_pixel_inf"], 2),
        l2_norm=round(def_res["delta_norm_l2"], 4),
    )

    total_time = (time.perf_counter() - total_start) * 1000.0

    margin_recovery_delta = def_res["semantic_margin"] - attack_res["semantic_margin"]

    summary = {
        "ground_truth": req.ground_truth,
        "clean_pred": clean_pred_label,
        "adv_pred": adv_pred_label,
        "def_pred": def_pred_label,
        "attack_success": attack_success,
        "defense_recovered": defense_recovered,
        "clean_margin": round(clean_margin, 4),
        "adv_margin": round(attack_res["semantic_margin"], 4),
        "def_margin": round(def_res["semantic_margin"], 4),
        "margin_gain": round(margin_recovery_delta, 4),
        "attack_budget_eps": req.epsilon_a,
        "defense_budget_eps": req.epsilon_d,
    }

    return PipelineResponse(
        clean=clean_stage,
        attacked=attack_stage,
        defended=defended_stage,
        attack_success=attack_success,
        defense_recovered=defense_recovered,
        ground_truth=req.ground_truth,
        total_time_ms=round(total_time, 1),
        summary=summary,
    )


@app.post("/api/evaluate/batch", response_model=BatchEvaluationResponse)
def evaluate_batch(
    dataset_preset: str = Form("cifar10"),
    epsilon_a: float = Form(4.0 / 255.0),
    steps_a: int = Form(10),
    epsilon_d: float = Form(4.0 / 255.0),
    steps_d: int = Form(5),
    warmup_steps: int = Form(5),
    num_views: int = Form(2),
):
    """Executes batch evaluation on curated dataset benchmark."""
    return run_batch_evaluation(
        engine=engine,
        dataset_preset=dataset_preset,
        epsilon_a=epsilon_a,
        steps_a=steps_a,
        epsilon_d=epsilon_d,
        steps_d=steps_d,
        warmup_steps=warmup_steps,
        num_views=num_views,
    )
