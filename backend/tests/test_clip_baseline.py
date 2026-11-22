"""Unit tests for CLIP ViT-B/32 baseline and feature encoding."""

import pytest
import torch
from PIL import Image

from scc.clip_model import get_clip_engine
from scc.sample_data import create_dog_image, CIFAR10_CLASSES


@pytest.fixture(scope="module")
def engine():
    return get_clip_engine()


def test_clip_engine_initialization(engine):
    assert engine.model is not None
    assert engine.tokenizer is not None
    assert engine.processor is not None
    # Model parameters must be frozen
    for param in engine.model.parameters():
        assert not param.requires_grad


def test_normalization_and_tensor_conversions(engine):
    dog_img = create_dog_image()
    tensor = engine.pil_to_tensor(dog_img)

    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.min() >= 0.0
    assert tensor.max() <= 1.0

    # Normalization should keep same spatial dimensions
    x_norm = engine.normalize(tensor)
    assert x_norm.shape == tensor.shape

    # Denormalization should recover values close to original
    x_rec = engine.denormalize(x_norm)
    assert torch.allclose(tensor, x_rec, atol=1e-5)


def test_zero_shot_encoding_and_classification(engine):
    dog_img = create_dog_image()
    img_tensor = engine.pil_to_tensor(dog_img)

    classes = CIFAR10_CLASSES
    prompts = [f"a photo of a {c}" for c in classes]
    text_feat = engine.encode_text(prompts, normalize=True)

    assert text_feat.shape == (len(classes), 512)
    # Norm of each text vector should be ~1.0
    norms = torch.norm(text_feat, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)

    # Encode image
    img_feat = engine.encode_image(img_tensor, normalize=True)
    assert img_feat.shape == (1, 512)
    assert torch.isclose(torch.norm(img_feat, p=2), torch.tensor(1.0), atol=1e-4)

    # Compute logits and probabilities
    logits, probs = engine.compute_logits_and_probs(img_feat, text_feat, use_temperature=True)
    assert logits.shape == (1, len(classes))
    assert probs.shape == (1, len(classes))
    assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-4)

    # Clean zero-shot prediction
    pred_idx = int(torch.argmax(logits, dim=-1).item())
    pred_class = classes[pred_idx]
    # Check that dog or a related animal is among top predictions
    assert pred_class in ["dog", "cat", "horse", "deer"]
