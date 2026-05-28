from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from api.schemas import (
    DashboardModelComparisonItem,
    DashboardModelsCompareResponse,
    DashboardRecommendationResponse,
    DashboardRecommendationScores,
    DashboardSummaryResponse,
)

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])

EXPERIMENT_NAME = "room_type_classification"
SCORING_VERSION = "v2"
EVAL_REPORTS_DIR = Path("./data/processed/evaluation_reports")


def _safe_float(value: Any):
    if value is None or value == "":
        return None
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None


def _to_iso8601_ms(timestamp_ms):
    if timestamp_ms is None:
        return None
    from datetime import datetime, timezone

    return datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).isoformat()


def _latest_comparison_csv(version_tag: str) -> Path | None:
    preferred = EVAL_REPORTS_DIR / f"model_comparison_{version_tag}.csv"
    if preferred.exists():
        return preferred
    candidates = sorted(EVAL_REPORTS_DIR.glob("model_comparison_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _load_models_from_csv(version_tag: str) -> list[dict[str, Any]]:
    csv_path = _latest_comparison_csv(version_tag)
    if csv_path is None:
        return []
    with csv_path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    models = []
    for row in rows:
        models.append(
            {
                "model": row.get("model"),
                "version_tag": row.get("version_tag") or version_tag,
                "accuracy": _safe_float(row.get("accuracy")),
                "f1_weighted": _safe_float(row.get("f1_weighted")),
                "latency_mean_ms": _safe_float(row.get("latency_mean_ms")),
                "throughput_img_per_sec": _safe_float(row.get("throughput_img_per_sec")),
                "cost_per_1k_images_dkk": _safe_float(row.get("cost_per_1k_dkk")),
                "maintainability_score": _safe_float(row.get("maintainability_score")),
                "param_count_M": _safe_float(row.get("param_count_M")),
            }
        )
    return models


def _latest_run_ids_by_model(experiment_name: str) -> tuple[dict[str, str], list[dict[str, Any]]]:
    try:
        import mlflow
    except ImportError:
        return {}, []
    client = mlflow.tracking.MlflowClient()
    try:
        experiment = client.get_experiment_by_name(experiment_name)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"MLflow unavailable: {exc}")
    if experiment is None:
        return {}, []
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=200,
    )
    latest = {}
    parsed_runs = []
    for run in runs:
        metrics = run.data.metrics
        params = run.data.params
        model_name = params.get("model")
        if model_name and model_name not in latest:
            latest[model_name] = run.info.run_id
        parsed_runs.append(
            {
                "run_id": run.info.run_id,
                "model": model_name,
                "version_tag": params.get("dataset_version_tag"),
                "scoring_version": params.get("scoring_version"),
                "start_time": run.info.start_time,
                "accuracy": _safe_float(metrics.get("final_accuracy")),
                "latency_mean_ms": _safe_float(metrics.get("latency_mean_ms")),
                "cost_per_1k_images_dkk": _safe_float(metrics.get("cost_per_1k_images_dkk")),
                "maintainability_score": _safe_float(metrics.get("maintainability_score")),
            }
        )
    return latest, parsed_runs


def _normalize_high(values: list[float | None]) -> dict[float, float]:
    clean = [v for v in values if v is not None]
    if not clean:
        return {}
    lo, hi = min(clean), max(clean)
    if hi == lo:
        return {v: 1.0 for v in clean}
    return {v: round((v - lo) / (hi - lo), 4) for v in clean}


def _normalize_low(values: list[float | None]) -> dict[float, float]:
    clean = [v for v in values if v is not None]
    if not clean:
        return {}
    lo, hi = min(clean), max(clean)
    if hi == lo:
        return {v: 1.0 for v in clean}
    return {v: round((hi - v) / (hi - lo), 4) for v in clean}


def _score_models(models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    quality_norm = _normalize_high([m.get("accuracy") for m in models])
    latency_norm = _normalize_low([m.get("latency_mean_ms") for m in models])
    cost_norm = _normalize_low([m.get("cost_per_1k_images_dkk") for m in models])
    maintain_norm = _normalize_high([m.get("maintainability_score") for m in models])

    weights = {
        "quality_score": 0.25,
        "latency_score": 0.25,
        "cost_score": 0.25,
        "maintainability_score": 0.25,
    }

    for model in models:
        q_raw = model.get("accuracy")
        l_raw = model.get("latency_mean_ms")
        c_raw = model.get("cost_per_1k_images_dkk")
        m_raw = model.get("maintainability_score")
        scores = {
            "quality_score": quality_norm.get(q_raw, 0.0),
            "latency_score": latency_norm.get(l_raw, 0.0),
            "cost_score": cost_norm.get(c_raw, 0.0),
            "maintainability_score": maintain_norm.get(m_raw, 0.0),
        }
        overall = sum(scores[k] * w for k, w in weights.items())
        model["scores"] = {k: round(v, 4) for k, v in scores.items()} | {"overall_score": round(overall, 4)}

    ranked = sorted(
        models,
        key=lambda x: (
            x["scores"]["overall_score"],
            x.get("accuracy") or 0,
            -(x.get("latency_mean_ms") or 10**9),
        ),
        reverse=True,
    )
    for idx, model in enumerate(ranked, start=1):
        model["recommendation_rank"] = idx
    return ranked


def _pick_best_model(runs: list[dict[str, Any]], metric_key: str, higher_is_better: bool = True):
    ranked = [r for r in runs if r.get(metric_key) is not None]
    if not ranked:
        return None
    ranked.sort(key=lambda r: r[metric_key], reverse=higher_is_better)
    return ranked[0].get("model")


def _reasoning(top: dict[str, Any], second: dict[str, Any] | None) -> list[str]:
    reasons = []
    if top.get("accuracy") is not None and top.get("latency_mean_ms") is not None:
        reasons.append("Best overall trade-off between accuracy and latency")
    if top.get("cost_per_1k_images_dkk") is not None:
        reasons.append("Competitive deployment cost among shortlisted models")
    if top.get("maintainability_score") is not None:
        reasons.append("Strong maintainability signal from a relatively small parameter footprint")
    if second and (top["scores"]["overall_score"] - second["scores"]["overall_score"]) >= 0.05:
        reasons[0] = "Clear overall choice across quality, latency, cost, and maintainability"
    return reasons[:3]


@router.get("/summary", response_model=DashboardSummaryResponse)
def dashboard_summary(
    experiment_name: str = Query(EXPERIMENT_NAME),
    dataset_version_tag: str = Query("cleaned_v1"),
):
    """
    Get the over-all evaluation summary for the candidate models, for instance, best model as per accuracy / latency / cost / maintainability.
    """
    
    latest_run_ids, parsed_runs = _latest_run_ids_by_model(experiment_name)
    if not parsed_runs:
        return DashboardSummaryResponse(
            experiment_name=experiment_name,
            dataset_version_tag=dataset_version_tag,
            scoring_version=SCORING_VERSION,
            total_runs=0,
            latest_run_time=None,
            best_model_by_accuracy=None,
            best_model_by_latency=None,
            best_model_by_cost=None,
            best_model_by_maintainability=None,
            recommended_model=None,
        )

    
    
    latest = parsed_runs[0]
    models = _load_models_from_csv(dataset_version_tag)
    ranked = _score_models(models) if models else []
    return DashboardSummaryResponse(
        experiment_name=experiment_name,
        dataset_version_tag=dataset_version_tag or latest.get("version_tag"),
        scoring_version=latest.get("scoring_version") or SCORING_VERSION,
        total_runs=len(parsed_runs),
        latest_run_time=_to_iso8601_ms(latest.get("start_time")),
        best_model_by_accuracy=_pick_best_model(parsed_runs, "accuracy", higher_is_better=True),
        best_model_by_latency=_pick_best_model(parsed_runs, "latency_mean_ms", higher_is_better=False),
        best_model_by_cost=_pick_best_model(parsed_runs, "cost_per_1k_images_dkk", higher_is_better=False),
        best_model_by_maintainability=_pick_best_model(parsed_runs, "maintainability_score", higher_is_better=True),
        recommended_model=ranked[0]["model"] if ranked else None,
    )


@router.get("/models/compare", response_model=DashboardModelsCompareResponse)
def compare_models(
    experiment_name: str = Query(EXPERIMENT_NAME),
    dataset_version_tag: str = Query("cleaned_v1"),
):
    """
    Compares candidate models across accuracy, latency, throughput, cost, and maintainability to rank which model offers the best trade-off for deployment.
    """
    models = _load_models_from_csv(dataset_version_tag)
    if not models:
        raise HTTPException(status_code=404, detail="No model comparison table found for the requested dataset version.")
    latest_run_ids, _ = _latest_run_ids_by_model(experiment_name)
    ranked = _score_models(models)
    return DashboardModelsCompareResponse(
        experiment_name=experiment_name,
        dataset_version_tag=dataset_version_tag,
        models=[
            DashboardModelComparisonItem(
                model=model["model"],
                latest_run_id=latest_run_ids.get(model["model"]),
                accuracy=model.get("accuracy"),
                f1_weighted=model.get("f1_weighted"),
                latency_mean_ms=model.get("latency_mean_ms"),
                throughput_img_per_sec=model.get("throughput_img_per_sec"),
                cost_per_1k_images_dkk=model.get("cost_per_1k_images_dkk"),
                maintainability_score=model.get("maintainability_score"),
                param_count_M=model.get("param_count_M"),
                recommendation_rank=model["recommendation_rank"],
            )
            for model in ranked
        ],
    )


@router.get("/recommendation", response_model=DashboardRecommendationResponse)
def recommendation(
    experiment_name: str = Query(EXPERIMENT_NAME),
    dataset_version_tag: str = Query("cleaned_v1"),
):
    """
    Get the final recommended model (and the runner up model) from the benchmark / evaluation results.
    """
    models = _load_models_from_csv(dataset_version_tag)
    if not models:
        raise HTTPException(status_code=404, detail="No model comparison table found for the requested dataset version.")
    ranked = _score_models(models)
    top = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else None
    return DashboardRecommendationResponse(
        recommended_model=top["model"],
        dataset_version_tag=dataset_version_tag,
        scoring_version=SCORING_VERSION,
        decision_rule="Balanced score using quality, latency, cost, and maintainability",
        reasoning=_reasoning(top, runner_up),
        #scores=DashboardRecommendationScores(**top["scores"]),
        runner_up_model=runner_up["model"] if runner_up else None,
#        build_vs_buy_recommendation="Build with the top ranked lightweight in-house model for this classification task; revisit buy options only if future latency SLAs or maintenance constraints become stricter.",
    )

