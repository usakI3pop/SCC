"""Untargeted White-Box Projected Gradient Descent (PGD) Attack on CLIP ViT-B/32.
Maximizes classification cross-entropy against ground-truth class while constraining perturbations within L-infinity ball.
"""

from __future__ import annotations

import time
from typing import Dict, Any, Optional

import torch
import torch.nn.functional as F

from .clip_model import CLIPEngine


def compute_semantic_margin(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    ground_truth_idx: int,
) -> float:
    """Computes the semantic margin m(x) = <f, t_y*> - max_{j != y*} <f, t_j> from Eq. (3) of paper."""
    # image_features: [1, d], text_features: [K, d]
    cos_sims = torch.matmul(image_features, text_features.T).squeeze(0)  # [K]
    target_sim = cos_sims[ground_truth_idx].item()

    num_classes = cos_sims.shape[0]
    competitor_indices = [i for i in range(num_classes) if i != ground_truth_idx]
    if competitor_indices:
        hardest_competitor_sim = cos_sims[competitor_indices].max().item()
    else:
        hardest_competitor_sim = target_sim

    margin = target_sim - hardest_competitor_sim
    return float(margin)


def pgd_attack(
    engine: CLIPEngine,
    clean_image: torch.Tensor,
    text_features: torch.Tensor,
    ground_truth_idx: int,
    epsilon: float = 4.0 / 255.0,
    alpha: Optional[float] = None,
    steps: int = 10,
    random_init: bool = True,
) -> Dict[str, Any]:
    """Untargeted white-box PGD attack maximizing classification cross-entropy against ground truth.

    Args:
        engine: CLIPEngine instance
        clean_image: Original image tensor [1, 3, 224, 224] in [0, 1]
        text_features: Precomputed normalized text embeddings [K, 512]
        ground_truth_idx: Index of true class in text_features
        epsilon: Maximum L-infinity perturbation budget
        alpha: Step size (defaults to 1.25 * eps / steps or 1/255)
        steps: Number of PGD iterations
        random_init: Whether to initialize with random uniform perturbation in [-eps, eps]

    Returns:
        Dict containing adversarial image, perturbation, norms, execution time, logits, probabilities, and semantic margin.
    """
    start_time = time.perf_counter()

    if alpha is None:
        alpha = max(1.0 / 255.0, (1.25 * epsilon) / max(steps, 1))

    # Initialize perturbation delta
    if random_init:
        delta = torch.empty_like(clean_image).uniform_(-epsilon, epsilon)
        # Ensure initial perturbed image is within [0, 1]
        delta = torch.clamp(clean_image + delta, 0.0, 1.0) - clean_image
    else:
        delta = torch.zeros_like(clean_image)

    delta.requires_grad = True

    target_tensor = torch.tensor([ground_truth_idx], dtype=torch.long, device=engine.device)

    # Iterative PGD steps
    for step in range(steps):
        perturbed_img = clean_image + delta
        # Forward pass through CLIP image encoder (differentiable)
        img_features = engine.encode_image(perturbed_img, normalize=True)
        logits, _ = engine.compute_logits_and_probs(img_features, text_features, use_temperature=True)

        # Cross-entropy loss against ground truth class
        # Maximizing this loss pushes prediction away from ground truth (untargeted attack)
        loss = F.cross_entropy(logits, target_tensor)

        # Compute gradient with respect to delta
        grad = torch.autograd.grad(loss, delta, retain_graph=False, create_graph=False)[0]

        # PGD ascent update with sign gradient
        delta_data = delta.data + alpha * grad.sign()

        # Project back into L-infinity ball [-epsilon, epsilon]
        delta_data = torch.clamp(delta_data, -epsilon, epsilon)

        # Project into valid pixel space [0, 1]
        delta_data = torch.clamp(clean_image + delta_data, 0.0, 1.0) - clean_image

        delta.data.copy_(delta_data)

    delta = delta.detach()
    adversarial_image = torch.clamp(clean_image + delta, 0.0, 1.0)

    # Final forward pass on adversarial image
    with torch.no_grad():
        adv_features = engine.encode_image(adversarial_image, normalize=True)
        adv_logits, adv_probs = engine.compute_logits_and_probs(adv_features, text_features, use_temperature=True)

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    # Perturbation metrics
    linf_norm = float(torch.norm(delta, p=float("inf")).item())
    l2_norm = float(torch.norm(delta, p=2).item())
    linf_pixel = float(linf_norm * 255.0)

    # Semantic margin under attack
    adv_margin = compute_semantic_margin(adv_features, text_features, ground_truth_idx)

    pred_idx = int(torch.argmax(adv_logits, dim=-1).item())
    attack_success = (pred_idx != ground_truth_idx)

    return {
        "adversarial_image": adversarial_image,
        "delta": delta,
        "linf_norm": linf_norm,
        "linf_pixel": linf_pixel,
        "l2_norm": l2_norm,
        "elapsed_ms": elapsed_ms,
        "logits": adv_logits.squeeze(0).cpu().numpy().tolist(),
        "probs": adv_probs.squeeze(0).cpu().numpy().tolist(),
        "pred_idx": pred_idx,
        "attack_success": attack_success,
        "semantic_margin": adv_margin,
    }
