"""
Model building, loading, preprocessing and forward pass.
Mirrors train_classifier.py so there is zero train/serve skew.
"""

import time
from io import BytesIO
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import (
    EfficientNet_B0_Weights,
    MobileNet_V3_Small_Weights,
    ResNet18_Weights,
    efficientnet_b0,
    mobilenet_v3_small,
    resnet18,
)

CLASSES = ["bathroom", "bedroom", "diningroom", "kitchen", "livingroom"]
IMG_SIZE = 224
MODELS_DIR = Path(__file__).parent.parent / "models"

MODEL_META = {
    "mobilenet_v3_small": {
        "param_count_M": 2.5,
        "cost_per_1k_dkk": 0.32,
        "maintainability_score": 5,
    },
    "efficientnet_b0": {
        "param_count_M": 5.3,
        "cost_per_1k_dkk": 0.51,
        "maintainability_score": 4,
    },
    "resnet18": {
        "param_count_M": 11.7,
        "cost_per_1k_dkk": 0.45,
        "maintainability_score": 5,
    },
}

_TRANSFORMS = transforms.Compose([
    transforms.Resize(int(IMG_SIZE * 1.14)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _build_backbone(name: str) -> nn.Module:
    num_classes = len(CLASSES)

    if name == "mobilenet_v3_small":
        model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif name == "efficientnet_b0":
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif name == "resnet18":
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unknown model: {name}")

    return model


def load_model(model_name: str, version_tag: str, device: torch.device) -> nn.Module:
    """Load a fine-tuned model from models/{model_name}_{version_tag}.pt"""
    weight_file = MODELS_DIR / f"{model_name}_{version_tag}.pt"
    if not weight_file.exists():
        raise FileNotFoundError(
            f"Weight file not found: {weight_file}. "
            f"Run train_classifier.py --version-tag {version_tag} first."
        )

    model = _build_backbone(model_name)
    state_dict = torch.load(str(weight_file), map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def list_available_models() -> list[dict]:
    """Scan models/ dir and return structured info for each .pt weight file."""
    available = []

    for pt_file in sorted(MODELS_DIR.glob("*.pt")):
        parts = pt_file.stem.split("_")
        try:
            cleaned_idx = next(i for i, part in enumerate(parts) if part == "cleaned")
            model_name = "_".join(parts[:cleaned_idx])
            version_tag = "_".join(parts[cleaned_idx:])
        except StopIteration:
            model_name = pt_file.stem
            version_tag = "unknown"

        meta = MODEL_META.get(
            model_name,
            {
                "param_count_M": 0.0,
                "cost_per_1k_dkk": 0.0,
                "maintainability_score": 0,
            },
        )

        available.append(
            {
                "model_name": model_name,
                "version_tag": version_tag,
                "weight_file": pt_file.name,
                "param_count_M": meta["param_count_M"],
                "cost_per_1k_dkk": meta["cost_per_1k_dkk"],
                "maintainability_score": meta["maintainability_score"],
            }
        )

    return available


@torch.no_grad()
def predict_image(image_bytes: bytes, model: nn.Module, device: torch.device) -> dict:
    """Run inference on raw image bytes. Returns class, confidence scores and timing."""
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    tensor = _TRANSFORMS(img).unsqueeze(0).to(device)

    t0 = time.perf_counter()
    logits = model(tensor)

    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()

    elapsed_ms = (time.perf_counter() - t0) * 1000
    probs = torch.softmax(logits, dim=1).squeeze().cpu().tolist()
    pred_idx = int(torch.argmax(logits, dim=1).item())

    return {
        "predicted_class": CLASSES[pred_idx],
        "confidence": round(probs[pred_idx], 4),
        "scores": [
            {"class_name": class_name, "confidence": round(prob, 4)}
            for class_name, prob in zip(CLASSES, probs)
        ],
        "inference_time_ms": round(elapsed_ms, 3),
    }
