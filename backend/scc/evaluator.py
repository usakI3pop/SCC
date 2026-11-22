"""Batch Evaluation Engine for Zero-Shot CLIP Robustness under PGD and SCC Defense.
Computes mathematically rigorous metrics:
- Clean Accuracy (%)
- Robust Accuracy under PGD (%)
- Defended Accuracy under SCC (%)
- Attack Success Rate (ASR %)
- Defense Recovery Rate (DRR %)
"""

from __future__ import annotations

import time
from typing import List, Dict, Any, Optional

import torch

from .clip_model import CLIPEngine
from .attacks import pgd_attack, compute_semantic_margin
from .defense import scc_defense
from .sample_data import get_sample_catalogue, get_sample_image
from .schemas import BatchSampleResult, BatchEvaluationResponse


def run_batch_evaluation(
    engine: CLIPEngine,
    dataset_preset: str = "cifar10",
    epsilon_a: float = 4.0 / 255.0,
    steps_a: int = 10,
    epsilon_d: float = 4.0 / 255.0,
    steps_d: int = 5,
    warmup_steps: int = 5,
    num_views: int = 2,
    lambda_cm: float = 4.0,
    temperature: float = 0.5,
    noise_sigma: float = 6.0 / 255.0,
) -> BatchEvaluationResponse:
    """Executes a complete batch evaluation run on benchmark samples using real PyTorch gradients."""
    start_total_time = time.perf_counter()

    catalogue = get_sample_catalogue()
    if dataset_preset != "all":
        catalogue = [s for s in catalogue if s.get("preset") == dataset_preset]
        if not catalogue:
            catalogue = get_sample_catalogue()

    results: List[BatchSampleResult] = []

    for item in catalogue:
        sample_id = item["id"]
        gt_label = item["ground_truth"]
        classes = item["classes"]

        # Ensure ground truth is among candidate classes
        if gt_label not in classes:
            classes = [gt_label] + [c for c in classes if c != gt_label]

        gt_idx = classes.index(gt_label)

        # Generate sample image
        pil_img = get_sample_image(sample_id)
        clean_tensor = engine.pil_to_tensor(pil_img)

        # Precompute candidate prompt embeddings
        prompts = [f"a photo of a {c}" for c in classes]
        text_features = engine.encode_text(prompts, normalize=True)

        # -------------------------------------------------------------
        # 1. Clean Zero-Shot CLIP
        # -------------------------------------------------------------
        with torch.no_grad():
            clean_feat = engine.encode_image(clean_tensor, normalize=True)
            clean_logits, _ = engine.compute_logits_and_probs(clean_feat, text_features, use_temperature=True)
            clean_pred_idx = int(torch.argmax(clean_logits, dim=-1).item())
            clean_pred_label = classes[clean_pred_idx]
            clean_correct = (clean_pred_idx == gt_idx)
            clean_margin = compute_semantic_margin(clean_feat, text_features, gt_idx)

        # -------------------------------------------------------------
        # 2. PGD Adversarial Attack
        # -------------------------------------------------------------
        attack_res = pgd_attack(
            engine=engine,
            clean_image=clean_tensor,
            text_features=text_features,
            ground_truth_idx=gt_idx,
            epsilon=epsilon_a,
            steps=steps_a,
            random_init=True,
        )
        adv_tensor = attack_res["adversarial_image"]
        adv_pred_idx = attack_res["pred_idx"]
        adv_pred_label = classes[adv_pred_idx]
        adv_correct = (adv_pred_idx == gt_idx)
        adv_margin = attack_res["semantic_margin"]
        attack_fooled = clean_correct and (not adv_correct)

        # -------------------------------------------------------------
        # 3. SCC Test-Time Defense
        # -------------------------------------------------------------
        def_res = scc_defense(
            engine=engine,
            adversarial_image=adv_tensor,
            text_features=text_features,
            ground_truth_idx=gt_idx,
            epsilon_d=epsilon_d,
            steps=steps_d,
            warmup_steps=warmup_steps,
            num_views=num_views,
            lambda_cm=lambda_cm,
            temperature=temperature,
            noise_sigma=noise_sigma,
        )
        def_pred_idx = def_res["pred_idx"]
        def_pred_label = classes[def_pred_idx]
        def_correct = (def_pred_idx == gt_idx)
        def_margin = def_res["semantic_margin"]
        defense_recovered = attack_fooled and def_correct

        results.append(
            BatchSampleResult(
                id=sample_id,
                name=item["name"],
                ground_truth=gt_label,
                clean_pred=clean_pred_label,
                clean_correct=clean_correct,
                clean_margin=clean_margin,
                adv_pred=adv_pred_label,
                adv_correct=adv_correct,
                adv_margin=adv_margin,
                attack_fooled=attack_fooled,
                def_pred=def_pred_label,
                def_correct=def_correct,
                def_margin=def_margin,
                defense_recovered=defense_recovered,
            )
        )

    total_time_ms = (time.perf_counter() - start_total_time) * 1000.0
    num_samples = len(results)

    clean_acc = (sum(1 for r in results if r.clean_correct) / max(num_samples, 1)) * 100.0
    robust_acc = (sum(1 for r in results if r.adv_correct) / max(num_samples, 1)) * 100.0
    defended_acc = (sum(1 for r in results if r.def_correct) / max(num_samples, 1)) * 100.0

    clean_correct_count = sum(1 for r in results if r.clean_correct)
    fooled_count = sum(1 for r in results if r.attack_fooled)
    recovered_count = sum(1 for r in results if r.defense_recovered)

    asr = (fooled_count / max(clean_correct_count, 1)) * 100.0 if clean_correct_count > 0 else 0.0
    drr = (recovered_count / max(fooled_count, 1)) * 100.0 if fooled_count > 0 else 0.0

    return BatchEvaluationResponse(
        total_samples=num_samples,
        clean_accuracy=round(clean_acc, 2),
        robust_accuracy=round(robust_acc, 2),
        defended_accuracy=round(defended_acc, 2),
        attack_success_rate=round(asr, 2),
        defense_recovery_rate=round(drr, 2),
        samples=results,
        total_time_ms=round(total_time_ms, 1),
    )
