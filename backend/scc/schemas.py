"""Pydantic schemas for the Adversarial Robustness and SCC Defense API."""

from __future__ import annotations

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class PredictionItem(BaseModel):
    label: str
    probability: float
    logit: float
    rank: int


class StageResult(BaseModel):
    stage_name: str
    image_base64: str
    predicted_label: str
    confidence: float
    is_correct: Optional[bool] = None
    semantic_margin: float
    top5_predictions: List[PredictionItem]
    execution_time_ms: float
    perturbation_base64: Optional[str] = None
    linf_norm: Optional[float] = None
    linf_pixel: Optional[float] = None
    l2_norm: Optional[float] = None


class ClassifyRequest(BaseModel):
    image_base64: Optional[str] = None
    sample_id: Optional[str] = None
    candidate_labels: List[str] = Field(default_factory=lambda: ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"])
    ground_truth: Optional[str] = None


class AttackRequest(BaseModel):
    image_base64: Optional[str] = None
    sample_id: Optional[str] = None
    candidate_labels: List[str]
    ground_truth: str
    epsilon: float = Field(default=4.0 / 255.0, description="L-infinity perturbation budget in [0, 1]")
    alpha: Optional[float] = Field(default=None, description="Step size per iteration")
    steps: int = Field(default=10, ge=1, le=50, description="Number of PGD iterations")
    random_init: bool = Field(default=True, description="Random uniform initialization inside L-inf ball")


class DefendRequest(BaseModel):
    image_base64: str = Field(description="Adversarial image in base64 format")
    candidate_labels: List[str]
    ground_truth: Optional[str] = None
    epsilon_d: float = Field(default=4.0 / 255.0, description="Corrective defense budget")
    alpha_d: float = Field(default=1.0 / 255.0, description="Defense step size")
    steps: int = Field(default=5, ge=1, le=20, description="SCC defense optimization steps")
    warmup_steps: int = Field(default=5, ge=0, le=10, description="TTC warm-up steps")
    num_views: int = Field(default=2, ge=1, le=4, description="Number of augmented views")
    lambda_cm: float = Field(default=4.0, description="Semantic consistency weight")
    temperature: float = Field(default=0.5, description="Sharpening temperature T")
    noise_sigma: float = Field(default=6.0 / 255.0, description="Augmentation Gaussian noise sigma")


class PipelineRequest(BaseModel):
    image_base64: Optional[str] = None
    sample_id: Optional[str] = None
    candidate_labels: List[str] = Field(default_factory=lambda: ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"])
    ground_truth: str = "dog"
    # Attack parameters
    epsilon_a: float = Field(default=4.0 / 255.0)
    steps_a: int = Field(default=10, ge=1, le=30)
    random_init: bool = Field(default=True)
    # Defense parameters
    epsilon_d: float = Field(default=4.0 / 255.0)
    steps_d: int = Field(default=5, ge=1, le=15)
    warmup_steps: int = Field(default=5, ge=0, le=10)
    num_views: int = Field(default=2, ge=1, le=4)
    lambda_cm: float = Field(default=4.0)
    temperature: float = Field(default=0.5)
    noise_sigma: float = Field(default=6.0 / 255.0)


class PipelineResponse(BaseModel):
    clean: StageResult
    attacked: StageResult
    defended: StageResult
    attack_success: bool
    defense_recovered: bool
    ground_truth: str
    total_time_ms: float
    summary: Dict[str, Any]


class BatchSampleResult(BaseModel):
    id: str
    name: str
    ground_truth: str
    clean_pred: str
    clean_correct: bool
    clean_margin: float
    adv_pred: str
    adv_correct: bool
    adv_margin: float
    attack_fooled: bool
    def_pred: str
    def_correct: bool
    def_margin: float
    defense_recovered: bool


class BatchEvaluationResponse(BaseModel):
    total_samples: int
    clean_accuracy: float
    robust_accuracy: float
    defended_accuracy: float
    attack_success_rate: float
    defense_recovery_rate: float
    samples: List[BatchSampleResult]
    total_time_ms: float
