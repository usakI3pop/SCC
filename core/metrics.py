"""Metrics and evaluation criteria for adversarial robustness in Vision-Language Models.

Implements:
1. Semantic Margin m(x) = <f, t_true> - max_{k != true} <f, t_k> from Eq. (3) of SCC paper.
2. L-infinity and L2 perturbation magnitudes.
3. Prediction correctness and Top-K distribution diagnostics.
"""

from __future__ import annotations

from typing import Dict, Any, List
import torch
import torch.nn.functional as F


def compute_semantic_margin(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    target_idx: int,
) -> float:
    """Computes the semantic margin m(x) from Eq. (3) of the SCC paper:
    
      m(x) = <f(x), t_{y*}> - max_{j != y*} <f(x), t_j>
      
    Args:
        image_features: Normalized image embedding [1, D].
        text_features: Normalized text embeddings [K, D].
        target_idx: Ground truth class index y*.
        
    Returns:
        Scalar margin float. Positive indicates correct alignment;
        negative indicates prediction has drifted to an incorrect competitor.
    """
    # Cosine similarities: [K]
    cos_sims = torch.matmul(image_features, text_features.T).squeeze(0)
    target_sim = cos_sims[target_idx].item()

    num_classes = cos_sims.shape[0]
    competitor_indices = [i for i in range(num_classes) if i != target_idx]
    if competitor_indices:
        hardest_competitor_sim = cos_sims[competitor_indices].max().item()
    else:
        hardest_competitor_sim = target_sim

    return float(target_sim - hardest_competitor_sim)


def compute_perturbation_norms(delta: torch.Tensor) -> Dict[str, float]:
    """Computes L-infinity and L2 norms of a perturbation tensor.
    
    Args:
        delta: Perturbation tensor in pixel space [1, 3, H, W].
        
    Returns:
        Dict with 'linf', 'linf_pixel' (scale 255), and 'l2'.
    """
    linf = float(torch.norm(delta.detach(), p=float("inf")).item())
    l2 = float(torch.norm(delta.detach(), p=2).item())
    return {
        "linf": linf,
        "linf_pixel": linf * 255.0,
        "l2": l2,
    }


def format_topk_predictions(
    logits: torch.Tensor,
    probs: torch.Tensor,
    class_names: List[str],
    k: int = 5,
) -> List[Dict[str, Any]]:
    """Formats top-K predictions with probabilities and class labels."""
    k = min(k, len(class_names))
    top_probs, top_indices = torch.topk(probs.squeeze(0), k=k)
    logits_sq = logits.squeeze(0)

    results = []
    for rank, (idx, prob) in enumerate(zip(top_indices.tolist(), top_probs.tolist()), start=1):
        results.append({
            "rank": rank,
            "class_idx": idx,
            "label": class_names[idx],
            "probability": float(prob),
            "logit": float(logits_sq[idx].item()),
        })
    return results
