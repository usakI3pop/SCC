"""Untargeted White-Box Projected Gradient Descent (PGD) Attack on CLIP.

Reference: Liu et al. (ICML 2026), Section 4.1 & Xing et al. (CVPR 2025).

Attack Specification:
- Threat Model: Untargeted white-box attack against undefended CLIP image encoder.
- Loss: Cross-entropy of CLIP logits against ground-truth class label (maximized).
- Optimization: Gradient ascent with sign update.
- Budget: L-infinity ball ||delta||_inf <= epsilon in [0, 1] pixel space.
- Projection: Clip delta to [-epsilon, epsilon], clamp (x + delta) to [0, 1].
- Defaults: epsilon = 1/255, 10 steps, alpha = 1/255, random_start = False.
"""

from __future__ import annotations

import time
from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .models import CLIPWrapper


def pgd_attack(
    model: CLIPWrapper,
    images: torch.Tensor,
    targets: torch.Tensor,
    class_names: Optional[List[str]] = None,
    text_features: Optional[torch.Tensor] = None,
    epsilon: float = 1.0 / 255.0,
    alpha: float = 1.0 / 255.0,
    steps: int = 10,
    random_start: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Executes untargeted white-box PGD attack on CLIP in true pixel space [0, 1].

    Args:
        model: Frozen CLIPWrapper module.
        images: Clean image batch [B, 3, 224, 224] in [0.0, 1.0].
        targets: Ground-truth class indices [B] in range [0, K-1].
        class_names: List of K class names (used if text_features is None).
        text_features: Precomputed normalized text embeddings [K, 512].
        epsilon: Maximum L-infinity perturbation budget (default: 1/255).
        alpha: Step size per iteration (default: 1/255).
        steps: Number of PGD iterations (default: 10).
        random_start: Whether to initialize with uniform noise in [-eps, eps] (default: False).

    Returns:
        Tuple containing:
          - adv_images: Adversarially perturbed images [B, 3, 224, 224] in [0, 1].
          - delta: Final perturbation tensor [B, 3, 224, 224].
          - diagnostics: Dict with timing, logits, probabilities, and attack metrics.
    """
    start_time = time.perf_counter()

    # Precompute or retrieve text embeddings
    if text_features is None:
        if class_names is None:
            raise ValueError("Either class_names or text_features must be provided.")
        text_features = model.encode_text_classes(class_names)

    device = images.device
    targets = targets.to(device)

    # Initialize perturbation delta in pixel space
    if random_start:
        delta = torch.empty_like(images).uniform_(-epsilon, epsilon)
        delta = torch.clamp(images + delta, 0.0, 1.0) - images
    else:
        delta = torch.zeros_like(images)

    delta.requires_grad_(True)

    # Iterative PGD gradient ascent steps
    for step in range(steps):
        adv_x = images + delta

        # Differentiable forward pass through image encoder and cosine similarity
        img_feats = model.encode_image(adv_x, normalize=True)
        logits = model.compute_logits(img_feats, text_features)

        # Cross-entropy loss against ground-truth:
        # Maximizing this pushes prediction away from the correct class
        loss = F.cross_entropy(logits, targets)

        # Compute exact gradient with respect to delta
        grad = torch.autograd.grad(loss, delta, retain_graph=False, create_graph=False)[0]

        with torch.no_grad():
            # Sign gradient ascent step
            delta_data = delta.data + alpha * grad.sign()

            # Project into L-infinity perturbation ball [-epsilon, epsilon]
            delta_data = torch.clamp(delta_data, -epsilon, epsilon)

            # Project into valid pixel range [0, 1]
            delta_data = torch.clamp(images + delta_data, 0.0, 1.0) - images

            delta.data.copy_(delta_data)

    delta = delta.detach()
    adv_images = torch.clamp(images + delta, 0.0, 1.0)

    # Final forward pass to evaluate adversarial prediction
    with torch.no_grad():
        adv_feats = model.encode_image(adv_images, normalize=True)
        adv_logits = model.compute_logits(adv_feats, text_features)
        adv_probs = F.softmax(adv_logits, dim=-1)

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    # Summary metrics
    preds = adv_logits.argmax(dim=-1)
    attack_success_mask = (preds != targets)
    asr = float(attack_success_mask.float().mean().item()) * 100.0
    robust_acc = 100.0 - asr

    linf_norm = float(torch.norm(delta, p=float("inf")).item())
    linf_pixel = linf_norm * 255.0
    l2_norm = float(torch.norm(delta.view(delta.size(0), -1), p=2, dim=1).mean().item())

    diagnostics = {
        "elapsed_ms": elapsed_ms,
        "linf_norm": linf_norm,
        "linf_pixel": linf_pixel,
        "l2_norm": l2_norm,
        "robust_acc": robust_acc,
        "asr": asr,
        "logits": adv_logits,
        "probs": adv_probs,
        "preds": preds,
    }

    return adv_images, delta, diagnostics
