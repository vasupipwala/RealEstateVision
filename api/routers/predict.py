"""POST /predict — upload an image and get a room type prediction."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from PIL import UnidentifiedImageError

from api.inference import load_model, predict_image
from api.routers.dashboard import _load_models_from_csv, _score_models
from api.dependencies import _state, require_model
from api.schemas import PredictResponse

router = APIRouter(prefix="/predict", tags=["Prediction"])

AVAILABLE_MODELS = ["mobilenet_v3_small", "efficientnet_b0", "resnet18"]


def _available_version_tags() -> list[str]:
    active_version = _state.get("version_tag")
    tags: list[str] = []

    for candidate in ("cleaned_v1", active_version):
        if candidate and candidate not in tags:
            tags.append(candidate)

    try:
        from api.routers.dashboard import EVAL_REPORTS_DIR

        if EVAL_REPORTS_DIR.exists():
            for path in sorted(EVAL_REPORTS_DIR.glob("model_comparison_*.csv")):
                tag = path.stem.replace("model_comparison_", "")
                if tag and tag not in tags:
                    tags.append(tag)
    except Exception:
        pass

    return tags


def _get_dashboard_recommendation() -> tuple[str | None, float | None]:
    version_tag = _state.get("version_tag") or "cleaned_v1"
    try:
        models = _load_models_from_csv(version_tag)
        if not models:
            return None, None
        ranked = _score_models(models)
        if not ranked:
            return None, None

        top = ranked[0]
        overall_score = None
        if isinstance(top.get("scores"), dict):
            overall_score = top["scores"].get("overall_score")

        return top.get("model"), overall_score
    except Exception:
        return None, None


def _build_model_description() -> str:
    #active_model = _state.get("model_name") or "unknown"
    recommended_model, recommended_score = _get_dashboard_recommendation()
    #active_model = recommended_model or "unknown"
    model_choices = ", ".join(f"'{model}'" for model in AVAILABLE_MODELS)

    if recommended_model:
        recommendation_text = (
            f"Recommended model: '{recommended_model}'."
#            + (
#                f" (dashboard overall score: {recommended_score:.4f})"
#                if recommended_score is not None
#                else ""
#            )
#            + "."
        )
    else:
        recommendation_text = "Recommended model: not available yet from dashboard scoring."

    return (
        f"""
        Override model for this request (Optional). 
        Available models: {model_choices}.
        {recommendation_text}
        """
    )


def _build_version_tag_description() -> str:
    active_version = _state.get("version_tag") or "unknown"
    version_choices = _available_version_tags()
    version_text = ", ".join(f"'{tag}'" for tag in version_choices) if version_choices else "none discovered"
    

    return (
        f"""
        Override Dataset version for this request (Optional).
        Available version tags: {version_text}.
        """
    )


@router.post("", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(..., description="JPEG, PNG, JPG, or WEBP room image."),
    model_name: str | None = Query(None, description=_build_model_description()),
    dataset_version_tag: str | None = Query(None, description=_build_version_tag_description()),
    state: dict = Depends(require_model),
):
    """
    Classify a room image into one of:\n
    `bathroom | bedroom | diningroom | kitchen | livingroom` \n
    Note: You may provide only model_name, only version_tag, or both.
    If omitted, the API uses the recommended model.
    """
    
    version_tag = dataset_version_tag

    if file.content_type not in {"image/jpeg", "image/png", "image/jpg", "image/webp"}:
        raise HTTPException(status_code=415, detail="Upload must be a JPEG, PNG, JPG, or WEBP image.")

    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    requested_model_name = model_name or state["model_name"]
    requested_version_tag = version_tag or state["version_tag"]

    if requested_model_name != state["model_name"] or requested_version_tag != state["version_tag"]:
        try:
            model = load_model(requested_model_name, requested_version_tag, state["device"])
            active_name = requested_model_name
            active_tag = requested_version_tag
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
    else:
        model = state["model"]
        active_name = state["model_name"]
        active_tag = state["version_tag"]

    try:
        result = predict_image(image_bytes, model, state["device"])
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=415,
            detail="Uploaded file could not be decoded as a supported image. Try exporting it as a standard JPEG or PNG.",
        )
    except OSError as exc:
        raise HTTPException(status_code=415, detail=f"Invalid image file: {exc}")
        
    
    return PredictResponse(
        predicted_class=result["predicted_class"],
        confidence=result["confidence"],
        scores=result["scores"],
        model_name=active_name,
        version_tag=active_tag,
        inference_time_ms=result["inference_time_ms"],
    )
