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
    param_count_M: float
    cost_per_1k_dkk: float
    maintainability_score: int


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
    accuracy: Optional[float]
    f1_weighted: Optional[float]
    f1_macro: Optional[float]
    mcc: Optional[float]
    latency_mean_ms: Optional[float]
    throughput_img_per_sec: Optional[float]
    cost_per_1k_images_dkk: Optional[float]
    maintainability_score: Optional[float]


class MLflowRun(BaseModel):
    run_id: str
    run_name: Optional[str]
    model: Optional[str]
    version_tag: Optional[str]
    git_commit: Optional[str] = None
    status: str
    start_time: Optional[str]
    metrics: RunMetrics


class RunsResponse(BaseModel):
    experiment_name: str
    runs: list[MLflowRun]


class HealthResponse(BaseModel):
    status: str
    db_ok: bool
    model_loaded: bool
    active_model: str
    device: str
    mlflow_ok: bool
    version: str
