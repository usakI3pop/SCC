"""API integration tests for FastAPI endpoints calling handler functions directly."""

import pytest
import main
from scc.sample_data import CIFAR10_CLASSES
from scc.schemas import ClassifyRequest, AttackRequest, DefendRequest, PipelineRequest


def test_api_health():
    res = main.health()
    assert res["status"] == "ready"
    assert "CLIP ViT-B/32" in res["model"]
    assert "Self-Calibrated Consistency" in res["framework"]


def test_api_samples():
    res = main.list_samples()
    assert "samples" in res
    assert len(res["samples"]) > 0
    assert "presets" in res
    assert "cifar10" in res["presets"]


def test_api_classify():
    req = ClassifyRequest(
        sample_id="dog",
        candidate_labels=["dog", "cat", "airplane"],
        ground_truth="dog",
    )
    res = main.classify(req)
    assert res.stage_name == "Clean CLIP Baseline"
    assert res.predicted_label in ["dog", "cat"]
    assert res.confidence > 0.0
    assert len(res.top5_predictions) >= 2
    assert res.image_base64.startswith("data:image/")


def test_api_pipeline():
    req = PipelineRequest(
        sample_id="dog",
        candidate_labels=["dog", "wolf", "cat", "airplane"],
        ground_truth="dog",
        epsilon_a=4.0 / 255.0,
        steps_a=2,
        epsilon_d=4.0 / 255.0,
        steps_d=2,
        warmup_steps=2,
        num_views=2,
    )
    res = main.full_pipeline(req)
    assert res.clean.stage_name == "Clean CLIP Baseline"
    assert res.attacked.stage_name == "PGD Adversarial Attack"
    assert res.defended.stage_name == "SCC Test-Time Defense"
    assert res.attacked.perturbation_base64.startswith("data:image/")
    assert res.defended.perturbation_base64.startswith("data:image/")
    assert "margin_gain" in res.summary
