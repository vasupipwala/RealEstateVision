"""GET /runs — list recent MLflow experiment runs with key metrics."""

from fastapi import APIRouter, HTTPException, Query

from api.schemas import MLflowRun, RunMetrics, RunsResponse

router = APIRouter(prefix="/runs", tags=["Experiment Tracking"])


def _safe_float(d: dict, key: str):
    value = d.get(key)
    return round(float(value), 4) if value is not None else None


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

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=limit,
    )

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
                status=run.info.status,
                start_time=str(run.info.start_time),
                metrics=RunMetrics(
                    accuracy=_safe_float(metrics, "final_accuracy"),
                    f1_weighted=_safe_float(metrics, "final_f1_weighted"),
                    f1_macro=_safe_float(metrics, "final_f1_macro"),
                    mcc=_safe_float(metrics, "final_mcc"),
                    latency_mean_ms=_safe_float(metrics, "latency_mean_ms"),
                    throughput_img_per_sec=_safe_float(metrics, "throughput_img_per_sec"),
                    cost_per_1k_images_dkk=_safe_float(metrics, "cost_per_1k_images_dkk"),
                    maintainability_score=_safe_float(metrics, "maintainability_score"),
                ),
            )
        )

    return RunsResponse(experiment_name=experiment_name, runs=results)
