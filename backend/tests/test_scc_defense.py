"""Unit tests for Self-Calibrated Consistency (SCC) test-time defense components."""

import pytest
import torch

from scc.clip_model import get_clip_engine
from scc.attacks import pgd_attack
from scc.defense import scc_defense, ttc_warmup, apply_view_augmentation
from scc.sample_data import create_dog_image, CIFAR10_CLASSES


@pytest.fixture(scope="module")
def engine():
    return get_clip_engine()


def test_view_augmentation(engine):
    dog_img = create_dog_image()
    tensor = engine.pil_to_tensor(dog_img)

    # View 0 (unflipped + noise)
    v0 = apply_view_augmentation(tensor, view_idx=0, noise_sigma=6.0 / 255.0)
    assert v0.shape == tensor.shape
    assert v0.min() >= 0.0 and v0.max() <= 1.0

    # View 1 (flipped + noise)
    v1 = apply_view_augmentation(tensor, view_idx=1, noise_sigma=6.0 / 255.0)
    assert v1.shape == tensor.shape
    assert v1.min() >= 0.0 and v1.max() <= 1.0


def test_ttc_warmup(engine):
    dog_img = create_dog_image()
    tensor = engine.pil_to_tensor(dog_img)

    eps_w = 4.0 / 255.0
    warmed = ttc_warmup(engine, tensor, warmup_steps=2, eps_w=eps_w, alpha_w=1.0 / 255.0)

    assert warmed.shape == tensor.shape
    assert warmed.min() >= 0.0 and warmed.max() <= 1.0
    delta = warmed - tensor
    max_d = torch.norm(delta, p=float("inf")).item()
    assert max_d <= eps_w + 1e-5


def test_scc_defense_execution(engine):
    dog_img = create_dog_image()
    clean_tensor = engine.pil_to_tensor(dog_img)

    classes = CIFAR10_CLASSES
    gt_idx = classes.index("dog")
    prompts = [f"a photo of a {c}" for c in classes]
    text_feat = engine.encode_text(prompts, normalize=True)

    # First attack with small steps
    attack_res = pgd_attack(
        engine=engine,
        clean_image=clean_tensor,
        text_features=text_feat,
        ground_truth_idx=gt_idx,
        epsilon=4.0 / 255.0,
        steps=2,
    )
    adv_tensor = attack_res["adversarial_image"]

    # Now defend with SCC
    eps_d = 4.0 / 255.0
    def_res = scc_defense(
        engine=engine,
        adversarial_image=adv_tensor,
        text_features=text_feat,
        ground_truth_idx=gt_idx,
        epsilon_d=eps_d,
        steps=2,
        warmup_steps=2,
        num_views=2,
        lambda_cm=4.0,
        temperature=0.5,
    )

    def_img = def_res["defended_image"]
    delta_def = def_res["delta"]

    # Defended image bounds
    assert def_img.min() >= 0.0 - 1e-6
    assert def_img.max() <= 1.0 + 1e-6

    # Perturbation bounds
    max_def = torch.norm(delta_def, p=float("inf")).item()
    assert max_def <= eps_d + 1e-5

    # Check probabilities shape and sum
    probs = def_res["probs"]
    assert len(probs) == len(classes)
    assert abs(sum(probs) - 1.0) < 1e-4

    # Check output dictionary keys
    assert "pseudo_label_idx" in def_res
    assert "semantic_margin" in def_res
    assert "elapsed_ms" in def_res
