"""GET /models — list available trained models. POST /models/load — swap active model."""

from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_state
from api.inference import list_available_models, load_model
from api.schemas import ModelInfo, ModelsResponse

router = APIRouter(prefix="/models", tags=["Models"])


@router.get("", response_model=ModelsResponse)
def list_models(state: dict = Depends(get_state)):
    """Return all available .pt weight files in models/ with metadata."""
    available = list_available_models()
    return ModelsResponse(
        available_models=[ModelInfo(**model) for model in available],
        active_model=state.get("model_name") or "none",
    )


@router.post("/load")
def load_model_endpoint(
    model_name: str,
    dataset_version_tag: str,
    state: dict = Depends(get_state),
):
    """
    Hot-swap the globally active model without restarting the server.
    e.g. POST /models/load?model_name=resnet18&version_tag=cleaned_v1
    """
    
    version_tag = dataset_version_tag
    
    try:
        model = load_model(model_name, version_tag, state["device"])
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    state["model"] = model
    state["model_name"] = model_name
    state["version_tag"] = version_tag

    return {
        "message": f"Model swapped to {model_name} ({version_tag})",
        "model_name": model_name,
        "version_tag": version_tag,
    }
