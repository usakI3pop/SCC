"""CLIP Model Wrapper and Text Embedding Cache using open_clip ViT-B/32 (OpenAI weights).

Strict Rules:
- Keep images in [0, 1] pixel space.
- CLIP mean/std normalization is applied INSIDE the wrapper differentiably so perturbations are in true pixel units.
- Text prompt: 'a photo of a {class}', L2-normalized, computed once and cached.
- Logits = model logit_scale.exp() * cosine_similarity.
- Weights frozen, model in eval mode.
"""

from __future__ import annotations

import os
import random
from typing import List, Tuple, Optional, Dict

# Ensure offline loading from local cache when available
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip

# Standard CLIP normalization constants (ViT-B/32)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def set_seed(seed: int = 42) -> None:
    """Sets seeds for torch, numpy, and random for deterministic behavior."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CLIPWrapper(nn.Module):
    """Wrapper around open_clip ViT-B/32 (OpenAI weights).
    
    Accepts raw pixel tensors in [0, 1], normalizes them internally,
    and returns L2-normalized visual embeddings or classification logits.
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: Optional[str] = None,
    ):
        super().__init__()
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load pretrained open_clip model and tokenizer
        self.model, _, _ = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=self.device,
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)

        # Put in eval mode and freeze all parameters
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        # Register normalization buffers (1, 3, 1, 1) on the target device
        self.register_buffer(
            "mean",
            torch.tensor(CLIP_MEAN, dtype=torch.float32, device=self.device).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor(CLIP_STD, dtype=torch.float32, device=self.device).view(1, 3, 1, 1),
        )

        # Text embedding cache: class_tuple -> normalized text feature tensor [K, D]
        self._text_cache: Dict[Tuple[str, ...], torch.Tensor] = {}

    def normalize(self, x_pixel: torch.Tensor) -> torch.Tensor:
        """Differentiably normalizes an image tensor from [0, 1] to CLIP input space."""
        return (x_pixel - self.mean) / self.std

    def denormalize(self, x_norm: torch.Tensor) -> torch.Tensor:
        """Inverse normalization back to [0, 1] pixel space."""
        return (x_norm * self.std + self.mean).clamp(0.0, 1.0)

    @property
    def logit_scale(self) -> torch.Tensor:
        """CLIP learnable temperature scaling factor tau = exp(logit_scale)."""
        return self.model.logit_scale.exp()

    def encode_image(self, x_pixel: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """Differentiably encodes pixel image tensor [B, 3, H, W] in [0, 1].
        
        Args:
            x_pixel: Float tensor in range [0, 1] with shape [B, 3, 224, 224].
            normalize: Whether to L2-normalize the output image embeddings.
            
        Returns:
            Image embeddings [B, 512].
        """
        x_norm = self.normalize(x_pixel)
        image_features = self.model.encode_image(x_norm)
        if normalize:
            image_features = F.normalize(image_features, p=2, dim=-1)
        return image_features

    def encode_text_classes(
        self,
        class_names: List[str],
        template: str = "a photo of a {class}",
    ) -> torch.Tensor:
        """Encodes class labels using prompt template and caches results.
        
        Args:
            class_names: List of K category strings.
            template: Prompt template string containing '{class}'.
            
        Returns:
            Normalized text feature tensor [K, 512].
        """
        cache_key = tuple(class_names)
        if cache_key in self._text_cache:
            return self._text_cache[cache_key]

        prompts = [template.format(c=c, **{"class": c}) for c in class_names]
        with torch.no_grad():
            text_tokens = self.tokenizer(prompts).to(self.device)
            text_features = self.model.encode_text(text_tokens)
            text_features = F.normalize(text_features, p=2, dim=-1)

        self._text_cache[cache_key] = text_features
        return text_features

    def compute_logits(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        """Computes CLIP classification logits = logit_scale * cosine_similarity.
        
        Args:
            image_features: Normalized visual features [B, 512].
            text_features: Normalized textual features [K, 512].
            
        Returns:
            Logits [B, K].
        """
        scale = self.logit_scale
        return scale * torch.matmul(image_features, text_features.T)

    def forward(
        self,
        x_pixel: torch.Tensor,
        class_names: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """End-to-end forward pass on raw pixel images in [0, 1].
        
        Args:
            x_pixel: Input image tensor [B, 3, 224, 224] in [0, 1].
            class_names: List of K candidate classes.
            
        Returns:
            Tuple of (logits [B, K], probabilities [B, K]).
        """
        img_feats = self.encode_image(x_pixel, normalize=True)
        txt_feats = self.encode_text_classes(class_names)
        logits = self.compute_logits(img_feats, txt_feats)
        probs = F.softmax(logits, dim=-1)
        return logits, probs
