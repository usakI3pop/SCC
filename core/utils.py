"""Utility functions for image processing, visualization, and perturbation rendering."""

from __future__ import annotations

import io
import base64
from typing import Tuple, Union
import numpy as np
from PIL import Image
import torch


def pil_to_tensor(
    img: Image.Image,
    target_size: Tuple[int, int] = (224, 224),
    device: Union[str, torch.device] = "cpu",
) -> torch.Tensor:
    """Converts a PIL image to a float tensor [1, 3, 224, 224] in [0, 1]."""
    rgb_img = img.convert("RGB").resize(target_size, Image.Resampling.BICUBIC)
    np_img = np.array(rgb_img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(np_img).permute(2, 0, 1).unsqueeze(0).to(device)
    return tensor


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Converts a float tensor [1, 3, H, W] in [0, 1] to a PIL Image."""
    t = tensor.detach().squeeze(0).permute(1, 2, 0).cpu().clamp(0.0, 1.0).numpy()
    np_img = (t * 255.0).round().astype(np.uint8)
    return Image.fromarray(np_img)


def tensor_to_base64(tensor: torch.Tensor, format: str = "PNG") -> str:
    """Encodes a [1, 3, H, W] tensor in [0, 1] to a base64 data URI."""
    pil_img = tensor_to_pil(tensor)
    buf = io.BytesIO()
    pil_img.save(buf, format=format)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{encoded}"


def render_amplified_perturbation(
    delta: torch.Tensor,
    amplification: float = 20.0,
) -> Image.Image:
    """Renders a perturbation tensor delta in [-eps, eps] amplified by x20.
    
    Perturbations are centered around 0.5 (neutral gray) to visualize
    both positive (brightened) and negative (darkened) high-frequency patterns.
    """
    d = delta.detach().squeeze(0).permute(1, 2, 0).cpu().numpy()
    vis = np.clip(d * amplification + 0.5, 0.0, 1.0)
    np_img = (vis * 255.0).round().astype(np.uint8)
    return Image.fromarray(np_img)
