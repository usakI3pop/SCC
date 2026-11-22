"""Unit tests for PGD adversarial attack."""

import pytest
import torch

from scc.clip_model import get_clip_engine
from scc.attacks import pgd_attack, compute_semantic_margin
from scc.sample_data import create_dog_image, CIFAR10_CLASSES


@pytest.fixture(scope="module")
def engine():
    return get_clip_engine()


def test_pgd_perturbation_bounds_and_clamping(engine):
    dog_img = create_dog_image()
    clean_tensor = engine.pil_to_tensor(dog_img)

    classes = CIFAR10_CLASSES
    gt_idx = classes.index("dog")
    prompts = [f"a photo of a {c}" for c in classes]
    text_feat = engine.encode_text(prompts, normalize=True)

    eps = 4.0 / 255.0
    res = pgd_attack(
        engine=engine,
        clean_image=clean_tensor,
        text_features=text_feat,
        ground_truth_idx=gt_idx,
        epsilon=eps,
        steps=3,
        random_init=True,
    )

    adv_img = res["adversarial_image"]
    delta = res["delta"]

    # 1. Output pixels must be clamped to [0, 1]
    assert adv_img.min() >= 0.0 - 1e-6
    assert adv_img.max() <= 1.0 + 1e-6

    # 2. Maximum perturbation must not exceed epsilon
    max_pert = torch.norm(delta, p=float("inf")).item()
    assert max_pert <= eps + 1e-5

    # 3. Check consistency of reported metrics
    assert abs(res["linf_norm"] - max_pert) < 1e-5
    assert res["l2_norm"] > 0.0

    # 4. Check semantic margin computation
    assert isinstance(res["semantic_margin"], float)


def test_semantic_margin_calculation(engine):
    classes = ["dog", "wolf", "cat"]
    prompts = [f"a photo of a {c}" for c in classes]
    text_feat = engine.encode_text(prompts, normalize=True)

    # Synthetic image feature aligned exactly with "dog"
    dog_feat = text_feat[0:1].clone()
    margin = compute_semantic_margin(dog_feat, text_feat, ground_truth_idx=0)
    # Cosine sim with dog is 1.0, with others < 1.0, so margin > 0
    assert margin > 0.0

    # Synthetic image feature aligned exactly with "wolf" (competitor)
    wolf_feat = text_feat[1:2].clone()
    margin_wolf = compute_semantic_margin(wolf_feat, text_feat, ground_truth_idx=0)
    # True class is dog (index 0), but image is wolf (index 1), so margin must be negative
    assert margin_wolf < 0.0
