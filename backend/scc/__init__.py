"""Self-Calibrated Consistency (SCC) and PGD Adversarial Robustness for Vision-Language Models.
Research reference: Liu et al., ICML 2026.
"""

from .clip_model import CLIPEngine, get_clip_engine
from .attacks import pgd_attack, compute_semantic_margin
from .defense import scc_defense, ttc_warmup

__all__ = [
    "CLIPEngine",
    "get_clip_engine",
    "pgd_attack",
    "compute_semantic_margin",
    "scc_defense",
    "ttc_warmup",
]
