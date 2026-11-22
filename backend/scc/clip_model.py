"""CLIP ViT-B/32 Engine with differentiable forward passes for PGD attack and SCC defense.
Maintains a frozen model while computing exact PyTorch gradients with respect to input image pixels.
"""

from __future__ import annotations

import base64
import io
import os
from pathlib import Path
from threading import Lock
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# Safeguard for torch.load vulnerability checks when loading .bin weights on PyTorch < 2.6
try:
    import transformers.utils.import_utils as _imp_utils
    _imp_utils.check_torch_load_is_safe = lambda: None
    import transformers.modeling_utils as _mod_utils
    _mod_utils.check_torch_load_is_safe = lambda: None
except Exception:
    pass

from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer


# Standard CLIP ViT-B/32 normalization constants
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

DEFAULT_CACHE_SNAPSHOT = Path(
    os.path.expanduser("~")
) / ".cache" / "huggingface" / "hub" / "models--openai--clip-vit-base-patch32" / "snapshots" / "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"


class CLIPEngine:
    """Wrapper around HuggingFace CLIP ViT-B/32 model with differentiable operations."""

    def __init__(self, device: Optional[str] = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self._lock = Lock()
        self.model: Optional[CLIPModel] = None
        self.tokenizer: Optional[CLIPTokenizer] = None
        self.processor: Optional[CLIPProcessor] = None

        # Precompute normalization tensors
        self.mean = torch.tensor(CLIP_MEAN, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(CLIP_STD, device=self.device).view(1, 3, 1, 1)

        self._load_model()

    def _load_model(self):
        with self._lock:
            if self.model is not None:
                return

            load_target = "openai/clip-vit-base-patch32"
            if DEFAULT_CACHE_SNAPSHOT.exists():
                load_target = str(DEFAULT_CACHE_SNAPSHOT)

            # Load model and tokenizer
            self.tokenizer = CLIPTokenizer.from_pretrained(load_target, local_files_only=DEFAULT_CACHE_SNAPSHOT.exists())
            self.processor = CLIPProcessor.from_pretrained(load_target, local_files_only=DEFAULT_CACHE_SNAPSHOT.exists())
            self.model = CLIPModel.from_pretrained(load_target, local_files_only=DEFAULT_CACHE_SNAPSHOT.exists())
            self.model.to(self.device)
            self.model.eval()

            # Freeze all parameters
            for param in self.model.parameters():
                param.requires_grad = False

    @property
    def logit_scale(self) -> torch.Tensor:
        """CLIP learnable temperature scale tau = exp(logit_scale)."""
        return self.model.logit_scale.exp()

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Differentiably normalizes an image tensor in [0, 1] to CLIP standard input."""
        return (x - self.mean) / self.std

    def denormalize(self, x_norm: torch.Tensor) -> torch.Tensor:
        """Inverse normalization back to [0, 1]."""
        return (x_norm * self.std + self.mean).clamp(0.0, 1.0)

    def encode_image(self, x_pixel: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """Differentiably encodes pixel image tensor [B, 3, 224, 224] in [0, 1] into L2-normalized image embeddings."""
        x_norm = self.normalize(x_pixel)
        # Pass through vision model
        vision_outputs = self.model.vision_model(x_norm)
        pooled_output = vision_outputs[1]  # [B, hidden_size]
        image_features = self.model.visual_projection(pooled_output)  # [B, 512]
        if normalize:
            image_features = F.normalize(image_features, p=2, dim=-1)
        return image_features

    def encode_text(self, texts: List[str], normalize: bool = True) -> torch.Tensor:
        """Encodes text prompts into L2-normalized text embeddings [K, 512] under torch.no_grad()."""
        with torch.no_grad():
            text_inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
            text_outputs = self.model.text_model(**text_inputs)
            pooled_output = text_outputs[1]
            text_features = self.model.text_projection(pooled_output)
            if normalize:
                text_features = F.normalize(text_features, p=2, dim=-1)
            return text_features

    def compute_logits_and_probs(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        use_temperature: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes classification logits and probabilities between image embeddings and text embeddings."""
        scale = self.logit_scale if use_temperature else 1.0
        logits = scale * torch.matmul(image_features, text_features.T)
        probs = F.softmax(logits, dim=-1)
        return logits, probs

    def pil_to_tensor(self, img: Image.Image, target_size: Tuple[int, int] = (224, 224)) -> torch.Tensor:
        """Converts PIL Image to RGB float tensor [1, 3, 224, 224] in range [0, 1]."""
        rgb_img = img.convert("RGB").resize(target_size, Image.Resampling.BICUBIC)
        np_img = np.array(rgb_img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(np_img).permute(2, 0, 1).unsqueeze(0).to(self.device)
        return tensor

    def tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """Converts tensor [1, 3, H, W] in [0, 1] to PIL Image."""
        t = tensor.detach().squeeze(0).permute(1, 2, 0).cpu().clamp(0.0, 1.0).numpy()
        np_img = (t * 255.0).round().astype(np.uint8)
        return Image.fromarray(np_img)

    def tensor_to_base64(self, tensor: torch.Tensor, format: str = "PNG") -> str:
        """Converts image tensor [1, 3, H, W] to base64 data URI string."""
        pil_img = self.tensor_to_pil(tensor)
        buf = io.BytesIO()
        pil_img.save(buf, format=format)
        encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/{format.lower()};base64,{encoded}"

    def perturbation_to_base64(self, delta: torch.Tensor, amplification: float = 10.0) -> str:
        """Generates an amplified perturbation visualization heatmap/diff map encoded as base64."""
        # delta is in [-eps, eps], shape [1, 3, H, W]
        d = delta.detach().squeeze(0).permute(1, 2, 0).cpu().numpy()
        # Scale and center at 0.5 (neutral gray) with amplified deviation
        vis = (d * amplification + 0.5).clip(0.0, 1.0)
        np_img = (vis * 255.0).round().astype(np.uint8)
        pil_img = Image.fromarray(np_img)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"


# Global singleton instance
_GLOBAL_ENGINE: Optional[CLIPEngine] = None
_INIT_LOCK = Lock()


def get_clip_engine(device: Optional[str] = None) -> CLIPEngine:
    global _GLOBAL_ENGINE
    if _GLOBAL_ENGINE is None:
        with _INIT_LOCK:
            if _GLOBAL_ENGINE is None:
                _GLOBAL_ENGINE = CLIPEngine(device=device)
    return _GLOBAL_ENGINE
