"""Core modules for Adversarial Robustness of VLMs using PGD and SCC."""

from .models import CLIPWrapper, set_seed, CLIP_MEAN, CLIP_STD
from .attacks import pgd_attack
from .metrics import compute_semantic_margin, compute_perturbation_norms, format_topk_predictions
from .utils import pil_to_tensor, tensor_to_pil, tensor_to_base64, render_amplified_perturbation

__all__ = [
    "CLIPWrapper",
    "set_seed",
    "CLIP_MEAN",
    "CLIP_STD",
    "pgd_attack",
    "compute_semantic_margin",
    "compute_perturbation_norms",
    "format_topk_predictions",
    "pil_to_tensor",
    "tensor_to_pil",
    "tensor_to_base64",
    "render_amplified_perturbation",
]
