"""POST /predict — upload an image and get a room type prediction."""

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile

from api.dependencies import require_model
from api.inference import load_model, predict_image
from api.schemas import PredictResponse

from PIL import UnidentifiedImageError


router = APIRouter(prefix="/predict", tags=["Prediction"])


@router.post("", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(..., description="JPEG or PNG room image"),
    model_name: str | None = Query(None, description="Override active model (e.g. resnet18)"),
    version_tag: str | None = Query(None, description="Override version tag (e.g. cleaned_v1)"),
    state: dict = Depends(require_model),
):
    """
    Classify a room image into one of:
    bathroom | bedroom | diningroom | kitchen | livingroom
    """
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg", "image/webp"):
        raise HTTPException(status_code=415, detail="Upload must be a JPEG, PNG, or WEBP image.")

    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if model_name and version_tag:
        try:
            model = load_model(model_name, version_tag, state["device"])
            active_name = model_name
            active_tag = version_tag
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
        raise HTTPException(
            status_code=415,
            detail=f"Invalid image file: {exc}",
        )


    return PredictResponse(
        predicted_class=result["predicted_class"],
        confidence=result["confidence"],
        scores=result["scores"],
        model_name=active_name,
        version_tag=active_tag,
        inference_time_ms=result["inference_time_ms"],
    )
