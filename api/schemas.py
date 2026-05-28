"""Pydantic response + request models for the RealEstateVision API."""

from typing import Optional

from pydantic import BaseModel, Field


class ClassScore(BaseModel):
    class_name: str
    confidence: float = Field(..., ge=0.0, le=1.0)


class PredictResponse(BaseModel):
    predicted_class: str
    confidence: float
    scores: list[ClassScore]
    model_name: str
    version_tag: str
    inference_time_ms: float


class ModelInfo(BaseModel):
    model_name: str
    version_tag: str
    weight_file: str
    accuracy: float
    latency_mean_ms: float
    #param_count_M: float
    cost_per_1k_dkk: float
    maintainability_score: float


class ModelsResponse(BaseModel):
    available_models: list[ModelInfo]
    active_model: str


class ValidationSummary(BaseModel):
    total_images: int
    valid_images: int
    corrupt_images: int
    blurry_images: int
    too_dark_images: int
    too_bright_images: int
    low_information_images: int
    odd_size_images: int
    exact_duplicate_images: int
    near_duplicate_images: int
    valid_ratio: float


class ValidationRecord(BaseModel):
    image_id: str
    raw_path: Optional[str]
    cleaned_path: Optional[str]
    split: Optional[str]
    class_name: Optional[str]
    is_corrupt: int
    is_blurry: int
    too_dark: int
    too_bright: int
    low_information: Optional[int] = None
    is_exact_duplicate: int
    is_near_duplicate: int
    duplicate_of: Optional[str]
    canonical_image_id: Optional[str]
    quality_score: Optional[float]
    is_valid: int
    drop_reason: Optional[str]
    validation_timestamp: Optional[str]


class DatasetVersion(BaseModel):
    id: int
    version_tag: str
    raw_image_count: Optional[int]
    cleaned_image_count: Optional[int]
    git_commit: Optional[str]
    created_at: Optional[str]
    notes: Optional[str]


class RunMetrics(BaseModel):
    accuracy: Optional[float] = None
    f1_weighted: Optional[float] = None
    f1_macro: Optional[float] = None
    mcc: Optional[float] = None
    cohen_kappa: Optional[float] = None
    top_2_accuracy: Optional[float] = None
    best_val_acc: Optional[float] = None
    best_epoch: Optional[float] = None
    latency_mean_ms: Optional[float] = None
    latency_p50_ms: Optional[float] = None
    latency_p95_ms: Optional[float] = None
    latency_p99_ms: Optional[float] = None
    throughput_img_per_sec: Optional[float] = None
    batch_throughput_img_per_sec: Optional[float] = None
    total_inference_sec: Optional[float] = None
    cost_per_1k_images_dkk: Optional[float] = None
    cost_for_test_set_dkk: Optional[float] = None
    cost_scale_10k_dkk: Optional[float] = None
    cost_scale_100k_dkk: Optional[float] = None
    cost_scale_1M_dkk: Optional[float] = None
    relative_compute_multiplier: Optional[float] = None
    baseline_cost_per_1k_dkk: Optional[float] = None
    baseline_throughput_img_per_sec: Optional[float] = None
    maintainability_score: Optional[float] = None
    param_count_M: Optional[float] = None
    trainable_params_M: Optional[float] = None
    trainable_params: Optional[int] = None
    dependency_count: Optional[int] = None
    maint_param_score_1_5: Optional[int] = None
    maint_trainable_score_1_5: Optional[int] = None
    maint_dependency_score_1_5: Optional[int] = None


class MLflowRun(BaseModel):
    run_id: str
    run_name: Optional[str]
    model: Optional[str]
    version_tag: Optional[str]
    git_commit: Optional[str] = None
    scoring_version: Optional[str] = None
    status: str
    start_time: Optional[str]
    metrics: RunMetrics


class RunsResponse(BaseModel):
    experiment_name: str
    runs: list[MLflowRun]


class DashboardSummaryResponse(BaseModel):
    experiment_name: str
    dataset_version_tag: Optional[str] = None
    scoring_version: Optional[str] = None
    total_runs: int
    latest_run_time: Optional[str] = None
    best_model_by_accuracy: Optional[str] = None
    best_model_by_latency: Optional[str] = None
    best_model_by_cost: Optional[str] = None
    best_model_by_maintainability: Optional[str] = None
    recommended_model: Optional[str] = None


class DashboardModelComparisonItem(BaseModel):
    model: str
    latest_run_id: Optional[str] = None
    accuracy: Optional[float] = None
    f1_weighted: Optional[float] = None
    latency_mean_ms: Optional[float] = None
    throughput_img_per_sec: Optional[float] = None
    cost_per_1k_images_dkk: Optional[float] = None
    maintainability_score: Optional[float] = None
    param_count_M: Optional[float] = None
    recommendation_rank: int


class DashboardModelsCompareResponse(BaseModel):
    experiment_name: str
    dataset_version_tag: str
    models: list[DashboardModelComparisonItem]


class DashboardRecommendationScores(BaseModel):
    quality_score: float
    latency_score: float
    cost_score: float
    maintainability_score: float
    overall_score: float


class DashboardRecommendationResponse(BaseModel):
    recommended_model: str
    dataset_version_tag: str
    scoring_version: str
    decision_rule: str
    reasoning: list[str]
    #scores: DashboardRecommendationScores
    runner_up_model: Optional[str] = None
    #build_vs_buy_recommendation: str


class HealthResponse(BaseModel):
    status: str
    db_ok: bool
    model_loaded: bool
    active_model: str
    device: str
    mlflow_ok: bool
    version: str
