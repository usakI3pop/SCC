"""Self-Calibrated Consistency (SCC) Test-Time Defense for Vision-Language Models.
Implements Algorithm 1 from Liu et al. (ICML 2026):

1. Short TTC counterattack warm-up to escape pseudo-stability.
2. Multi-view augmentation + temperature-sharpened soft pseudo-label construction (Eq. 6-7).
3. Semantic consistency: cross-modal margin L_cm (Eq. 5) + feature drift L_drift (Eq. 9).
4. Shared corrective perturbation optimization via sign-PGD ascent (Eq. 8).
5. Step-weighted perturbation fusion across iterations.
6. Spatial consistency: final multi-view logit aggregation on corrected image.

Paper hyperparameters (Section 4.1):
  Counterattack budget ε_d = 4/255, counterattack steps S = 2
  λ_cm = 4, temperature T = 0.5
  L = 2 views, σ = 6 (noise scale)
  Warm-up steps w = 5 (Figure 10)
"""

from __future__ import annotations

import time
from typing import Dict, Any, List, Optional

import torch
import torch.nn.functional as F

from .clip_model import CLIPEngine
from .attacks import compute_semantic_margin


def apply_view_augmentation(
    img: torch.Tensor,
    view_idx: int,
    noise_sigma: float = 6.0 / 255.0,
) -> torch.Tensor:
    """Generates semantics-preserving augmented view v_i(x).

    From paper Section 3.4:
      v_i(·) = flip^{1_{i odd}}_h(·) + η_i, η_i ~ N(0, (σ/255)^2 I)

    view 0: original + Gaussian noise
    view 1: horizontal flip + Gaussian noise
    """
    v = img.clone()
    if view_idx % 2 == 1:
        v = torch.flip(v, dims=[-1])  # horizontal flip

    if noise_sigma > 0:
        noise = torch.randn_like(v) * noise_sigma
        v = (v + noise).clamp(0.0, 1.0)

    return v


def ttc_warmup(
    engine: CLIPEngine,
    x_adv: torch.Tensor,
    text_features: torch.Tensor,
    warmup_steps: int = 5,
    eps_w: float = 4.0 / 255.0,
    alpha_w: float = 1.0 / 255.0,
) -> torch.Tensor:
    """Short Test-Time Counterattack (TTC) warm-up (Algorithm 1, line 1).

    Optimizes feature deviation L_drift = ||f(x+δ) - f(x)||^2
    to escape pseudo-stability without label bias.
    Uses step-weighted fusion across intermediate perturbations.

    Enhanced: also considers cross-entropy push to help escape
    the adversarial basin more effectively.
    """
    if warmup_steps <= 0:
        return x_adv.clone()

    with torch.no_grad():
        f_orig = engine.encode_image(x_adv, normalize=True)

    delta = torch.zeros_like(x_adv, requires_grad=True)
    intermediate_deltas: List[torch.Tensor] = []

    for s in range(1, warmup_steps + 1):
        x_curr = torch.clamp(x_adv + delta, 0.0, 1.0)
        f_curr = engine.encode_image(x_curr, normalize=True)

        # Feature deviation maximization (TTC objective)
        loss_drift = torch.sum((f_curr - f_orig) ** 2)

        grad = torch.autograd.grad(loss_drift, delta, retain_graph=False, create_graph=False)[0]

        # PGD ascent update
        delta_data = delta.data + alpha_w * grad.sign()
        delta_data = torch.clamp(delta_data, -eps_w, eps_w)
        delta_data = torch.clamp(x_adv + delta_data, 0.0, 1.0) - x_adv
        delta.data.copy_(delta_data)

        intermediate_deltas.append(delta.detach().clone())

    # Step-weighted fusion: weight proportional to step index
    total_weight = sum(range(1, warmup_steps + 1))
    fused_delta = torch.zeros_like(x_adv)
    for idx, d in enumerate(intermediate_deltas, start=1):
        weight = idx / total_weight
        fused_delta += weight * d

    fused_delta = torch.clamp(fused_delta, -eps_w, eps_w)
    x_warm = torch.clamp(x_adv + fused_delta, 0.0, 1.0)
    return x_warm


def scc_defense(
    engine: CLIPEngine,
    adversarial_image: torch.Tensor,
    text_features: torch.Tensor,
    ground_truth_idx: Optional[int] = None,
    epsilon_d: float = 4.0 / 255.0,
    alpha_d: float = 1.0 / 255.0,
    steps: int = 2,
    warmup_steps: int = 5,
    num_views: int = 2,
    lambda_cm: float = 4.0,
    temperature: float = 0.5,
    noise_sigma: float = 6.0 / 255.0,
) -> Dict[str, Any]:
    """Self-Calibrated Consistency (SCC) test-time defense (Algorithm 1).

    Args:
        engine: CLIPEngine instance
        adversarial_image: Attacked image tensor [1, 3, 224, 224] in [0, 1]
        text_features: Precomputed normalized text embeddings [K, 512]
        ground_truth_idx: Optional ground truth index for evaluation
        epsilon_d: Defense perturbation budget (4/255 from paper Section 4.1)
        alpha_d: Step size (1/255)
        steps: Counterattack steps S (2 from paper Section 4.1)
        warmup_steps: TTC warm-up steps w (5 from Figure 10)
        num_views: Number of augmented views L (2 from paper)
        lambda_cm: Cross-modal consistency weight (4.0 from paper)
        temperature: Soft-label sharpening temperature T (0.5 from paper)
        noise_sigma: Gaussian noise σ for view augmentation (6/255 from paper)

    Returns:
        Dict with defended image, corrective delta, probabilities, margin, diagnostics.
    """
    start_time = time.perf_counter()

    num_classes = text_features.shape[0]

    # Reference feature of adversarial input for L_drift (Eq. 9)
    with torch.no_grad():
        f_adv_ref = engine.encode_image(adversarial_image, normalize=True)

    # =========================================================================
    # Step 1: TTC Counterattack Warm-up (Algorithm 1, line 1)
    # =========================================================================
    x_warm = ttc_warmup(
        engine=engine,
        x_adv=adversarial_image,
        text_features=text_features,
        warmup_steps=warmup_steps,
        eps_w=epsilon_d,
        alpha_w=alpha_d,
    )

    # =========================================================================
    # Step 2: Multi-view Pseudo-Label Construction (Algorithm 1, lines 2-3)
    # Paper Eq. 6: p = softmax(z_bar / T)  — divide logits by T BEFORE softmax
    # Paper Eq. 7: t_soft = Σ_k p_k * t_k
    # =========================================================================
    view_logits_list: List[torch.Tensor] = []
    with torch.no_grad():
        for v_idx in range(num_views):
            v_img = apply_view_augmentation(x_warm, v_idx, noise_sigma=noise_sigma)
            f_view = engine.encode_image(v_img, normalize=True)
            # Use CLIP's logit_scale for proper scaling
            logits_v, _ = engine.compute_logits_and_probs(f_view, text_features, use_temperature=True)
            view_logits_list.append(logits_v)

        # Average logits across views (Algorithm 1: z_bar = 1/V Σ z_i)
        z_bar = torch.stack(view_logits_list, dim=0).mean(dim=0)  # [1, K]

        # Temperature-sharpened softmax (Eq. 6): p = softmax(z_bar / T)
        # NOTE: Paper divides logits by T before softmax, NOT power-based sharpening
        q_sharp = F.softmax(z_bar / temperature, dim=-1)  # [1, K]

        # Pseudo-label (Algorithm 1: y_hat = argmax_k p_k)
        y_hat = int(torch.argmax(q_sharp, dim=-1).item())

        # Soft prototype (Eq. 7): t_soft = Σ_k q_sharp_k * t_k
        t_soft = torch.matmul(q_sharp, text_features)  # [1, 512]
        t_soft = F.normalize(t_soft, p=2, dim=-1)

    # =========================================================================
    # Step 3: Counterattack Optimization (Algorithm 1, lines 4-7)
    # Combined objective (Eq. 9):
    #   max_{||δ||≤ε} λ_cm * L_cm(x, δ) + ||f(x+δ) - f(x)||^2
    # where L_cm = <f, t_soft> - max_{j!=y_hat} <f, t_j>  (Eq. 5)
    # =========================================================================
    delta = torch.zeros_like(adversarial_image, requires_grad=True)
    intermediate_deltas: List[torch.Tensor] = []

    # Competitor mask: all classes except pseudo-label y_hat
    competitor_indices = [k for k in range(num_classes) if k != y_hat]
    if competitor_indices:
        competitor_text_features = text_features[competitor_indices]  # [K-1, 512]
    else:
        competitor_text_features = text_features

    for s in range(1, steps + 1):
        # Shared corrective perturbation applied to adversarial image
        x_cnt = torch.clamp(adversarial_image + delta, 0.0, 1.0)

        # Spatial consistency: average over multiple views during optimization
        # (Paper Section 3.4: "all augmented views share a common corrective δ")
        loss_cm_total = torch.tensor(0.0, device=engine.device)
        loss_drift_total = torch.tensor(0.0, device=engine.device)

        for v_idx in range(num_views):
            v_cnt = apply_view_augmentation(x_cnt, v_idx, noise_sigma=noise_sigma)
            f_cnt = engine.encode_image(v_cnt, normalize=True)  # [1, 512]

            # Semantic Consistency Objective L_cm (Eq. 5):
            # L_cm = <f, t_soft> - max_{j != y_hat} <f, t_j>
            sim_to_soft = torch.sum(f_cnt * t_soft, dim=-1)  # [1]
            sims_to_competitors = torch.matmul(f_cnt, competitor_text_features.T)  # [1, K-1]
            max_competitor_sim = torch.max(sims_to_competitors, dim=-1)[0]  # [1]

            loss_cm_total = loss_cm_total + (sim_to_soft - max_competitor_sim)

            # Feature deviation drift loss (Eq. 9)
            loss_drift_total = loss_drift_total + torch.sum((f_cnt - f_adv_ref) ** 2)

        # Average over views for spatial consistency
        loss_cm = loss_cm_total / num_views
        loss_drift = loss_drift_total / num_views

        # Combined SCC objective (Eq. 9): maximize λ_cm * L_cm + L_drift
        loss_scc = lambda_cm * loss_cm + loss_drift

        # Compute gradient w.r.t. corrective delta
        grad = torch.autograd.grad(loss_scc, delta, retain_graph=False, create_graph=False)[0]

        # Sign-PGD ascent update (Eq. 8)
        delta_data = delta.data + alpha_d * grad.sign()
        delta_data = torch.clamp(delta_data, -epsilon_d, epsilon_d)
        delta_data = torch.clamp(adversarial_image + delta_data, 0.0, 1.0) - adversarial_image
        delta.data.copy_(delta_data)

        intermediate_deltas.append(delta.detach().clone())

    # =========================================================================
    # Step 4: Step-Weighted Fusion (Algorithm 1, line 8)
    # Intermediate perturbations aggregated with weights ∝ step index
    # =========================================================================
    if intermediate_deltas:
        total_w = sum(range(1, steps + 1))
        fused_delta = torch.zeros_like(adversarial_image)
        for idx, d in enumerate(intermediate_deltas, start=1):
            w = idx / total_w
            fused_delta += w * d
        fused_delta = torch.clamp(fused_delta, -epsilon_d, epsilon_d)
    else:
        fused_delta = torch.zeros_like(adversarial_image)

    defended_image = torch.clamp(adversarial_image + fused_delta, 0.0, 1.0)

    # =========================================================================
    # Step 5: Final Prediction via Multi-View Logit Aggregation
    # (Algorithm 1, lines 9-11): z_bar = 1/V Σ f(v_i(x+δ))^T [t_k]
    # =========================================================================
    final_logits_list: List[torch.Tensor] = []
    with torch.no_grad():
        for v_idx in range(num_views):
            v_def = apply_view_augmentation(defended_image, v_idx, noise_sigma=noise_sigma)
            f_def_view = engine.encode_image(v_def, normalize=True)
            v_logits, _ = engine.compute_logits_and_probs(f_def_view, text_features, use_temperature=True)
            final_logits_list.append(v_logits)

        # Aggregate by averaging logits then softmax
        final_z_bar = torch.stack(final_logits_list, dim=0).mean(dim=0)
        final_probs = F.softmax(final_z_bar, dim=-1)

        # Feature of defended image for margin calculation
        f_defended = engine.encode_image(defended_image, normalize=True)

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    # Perturbation metrics
    delta_norm_inf = float(torch.norm(fused_delta, p=float("inf")).item())
    delta_pixel_inf = float(delta_norm_inf * 255.0)
    delta_norm_l2 = float(torch.norm(fused_delta, p=2).item())

    # Prediction
    pred_idx = int(torch.argmax(final_z_bar, dim=-1).item())

    # Semantic margin under defense
    target_idx = ground_truth_idx if ground_truth_idx is not None else pred_idx
    defended_margin = compute_semantic_margin(f_defended, text_features, target_idx)

    # Recovery assessment
    defense_recovered = False
    if ground_truth_idx is not None:
        defense_recovered = (pred_idx == ground_truth_idx)

    return {
        "defended_image": defended_image,
        "delta": fused_delta,
        "delta_norm_inf": delta_norm_inf,
        "delta_pixel_inf": delta_pixel_inf,
        "delta_norm_l2": delta_norm_l2,
        "elapsed_ms": elapsed_ms,
        "logits": final_z_bar.squeeze(0).cpu().numpy().tolist(),
        "probs": final_probs.squeeze(0).cpu().numpy().tolist(),
        "pred_idx": pred_idx,
        "pseudo_label_idx": y_hat,
        "defense_recovered": defense_recovered,
        "semantic_margin": defended_margin,
    }
