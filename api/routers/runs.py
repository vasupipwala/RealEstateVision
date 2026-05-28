"""GET /runs — list recent MLflow experiment runs with key metrics."""

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query

from api.schemas import MLflowRun, RunMetrics, RunsResponse

router = APIRouter(prefix="/runs", tags=["Experiment Tracking"])


def _safe_float(d: dict, key: str):
    value = d.get(key)
    return round(float(value), 4) if value is not None else None


def _safe_int(d: dict, key: str):
    value = d.get(key)
    return int(value) if value is not None else None


def _to_iso8601_ms(timestamp_ms):
    if timestamp_ms is None:
        return None
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).isoformat()


@router.get("", response_model=RunsResponse)
def list_runs(
    experiment_name: str = Query("room_type_classification"),
    limit: int = Query(20, ge=1, le=100),
):
    """Recent MLflow runs with quality, performance, cost and maintainability metrics."""
    try:
        import mlflow
    except ImportError:
        raise HTTPException(status_code=503, detail="mlflow is not installed.")

    client = mlflow.tracking.MlflowClient()

    try:
        experiment = client.get_experiment_by_name(experiment_name)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"MLflow unavailable: {exc}")

    if experiment is None:
        return RunsResponse(experiment_name=experiment_name, runs=[])

    try:
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=limit,
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Failed to fetch runs from MLflow: {exc}")

    results = []
    for run in runs:
        metrics = run.data.metrics
        params = run.data.params

        results.append(
            MLflowRun(
                run_id=run.info.run_id,
                run_name=run.info.run_name,
                model=params.get("model"),
                version_tag=params.get("dataset_version_tag"),
                git_commit=params.get("git_commit"),
                scoring_version=params.get("scoring_version"),
                status=run.info.status,
                start_time=_to_iso8601_ms(run.info.start_time),
                metrics=RunMetrics(
                    accuracy=_safe_float(metrics, "final_accuracy"),
                    f1_weighted=_safe_float(metrics, "final_f1_weighted"),
                    f1_macro=_safe_float(metrics, "final_f1_macro"),
                    mcc=_safe_float(metrics, "final_mcc"),
                    cohen_kappa=_safe_float(metrics, "final_cohen_kappa"),
                    top_2_accuracy=_safe_float(metrics, "final_top2_accuracy"),
                    best_val_acc=_safe_float(metrics, "best_val_acc"),
                    best_epoch=_safe_float(metrics, "best_epoch"),
                    latency_mean_ms=_safe_float(metrics, "latency_mean_ms"),
                    latency_p50_ms=_safe_float(metrics, "latency_p50_ms"),
                    latency_p95_ms=_safe_float(metrics, "latency_p95_ms"),
                    latency_p99_ms=_safe_float(metrics, "latency_p99_ms"),
                    throughput_img_per_sec=_safe_float(metrics, "throughput_img_per_sec"),
                    batch_throughput_img_per_sec=_safe_float(metrics, "batch_throughput_img_per_sec"),
                    total_inference_sec=_safe_float(metrics, "total_inference_sec"),
                    cost_per_1k_images_dkk=_safe_float(metrics, "cost_per_1k_images_dkk"),
                    cost_for_test_set_dkk=_safe_float(metrics, "cost_for_test_set_dkk"),
                    cost_scale_10k_dkk=_safe_float(metrics, "cost_scale_10k_dkk"),
                    cost_scale_100k_dkk=_safe_float(metrics, "cost_scale_100k_dkk"),
                    cost_scale_1M_dkk=_safe_float(metrics, "cost_scale_1M_dkk"),
                    relative_compute_multiplier=_safe_float(metrics, "relative_compute_multiplier"),
                    baseline_cost_per_1k_dkk=_safe_float(metrics, "baseline_cost_per_1k_dkk"),
                    baseline_throughput_img_per_sec=_safe_float(metrics, "baseline_throughput_img_per_sec"),
                    maintainability_score=_safe_float(metrics, "maintainability_score"),
                    param_count_M=_safe_float(metrics, "param_count_M"),
                    trainable_params_M=_safe_float(metrics, "trainable_params_M"),
                    trainable_params=_safe_int(metrics, "trainable_params"),
                    dependency_count=_safe_int(metrics, "dependency_count"),
                    maint_param_score_1_5=_safe_int(metrics, "maint_param_score_1_5"),
                    maint_trainable_score_1_5=_safe_int(metrics, "maint_trainable_score_1_5"),
                    maint_dependency_score_1_5=_safe_int(metrics, "maint_dependency_score_1_5"),
                ),
            )
        )

    return RunsResponse(experiment_name=experiment_name, runs=results)
