"""
Model building, loading, preprocessing and forward pass.
Mirrors train_classifier.py so there is zero train/serve skew.
Reads model metadata from evaluation report artifacts instead of hard-coded values.
"""

import json
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

ROOT_DIR = Path(__file__).parent.parent
MODELS_DIR = ROOT_DIR / "models"
EVAL_REPORTS_DIR = ROOT_DIR / "data" / "processed" / "evaluation_reports"

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


def _parse_model_file_name(pt_file: Path) -> tuple[str, str]:
    """
    Parse model and version from filenames like:
    mobilenet_v3_small_cleaned_v1.pt
    """
    parts = pt_file.stem.split("_")
    try:
        cleaned_idx = next(i for i, part in enumerate(parts) if part == "cleaned")
        model_name = "_".join(parts[:cleaned_idx])
        version_tag = "_".join(parts[cleaned_idx:])
    except StopIteration:
        model_name = pt_file.stem
        version_tag = "unknown"

    return model_name, version_tag


def _default_model_meta() -> dict:
    return {
        "accuracy": 0.0,
        "latency_mean_ms": 0.0,
        #"param_count_M": 0.0,
        "cost_per_1k_dkk": 0.0,
        "maintainability_score": 0.0,
    }


def _load_model_meta(model_name: str, version_tag: str) -> dict:
    """
    Load computed metadata from:
    data/processed/evaluation_reports/{model_name}_{version_tag}_report.json

    Expected fields come from the updated train_classifier.py report:
    - report["quality"]["accuracy"]
    - report["performance"]["latencymeanms"]
    - report["cost"]["costper1kimagesdkk"]
    - report["maintainability"]["maintainabilityscore"]
    """
    report_path = EVAL_REPORTS_DIR / f"{model_name}_{version_tag}_report.json"

    if not report_path.exists():
        return _default_model_meta()

    try:
        report = json.loads(report_path.read_text())
    except Exception:
        return _default_model_meta()
        
    
    quality = report.get("quality", {})
    performance = report.get("performance", {})
    cost = report.get("cost", {})
    maintainability = report.get("maintainability", {})

    return {
        "accuracy": float(quality.get("accuracy", 0.0)),
        "latency_mean_ms": float(performance.get("latency_mean_ms", 0.0)),
        #"param_count_M": float(maintainability.get("param_count_M", 0.0)),
        "cost_per_1k_dkk": float(cost.get("cost_per_1k_images_dkk", 0.0)),
        "maintainability_score": float(maintainability.get("maintainability_score", 0.0)),
    }


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
        model_name, version_tag = _parse_model_file_name(pt_file)
        meta = _load_model_meta(model_name, version_tag)

        available.append(
            {
                "model_name": model_name,
                "version_tag": version_tag,
                "weight_file": pt_file.name,
                "accuracy": round(meta["accuracy"], 3),
                "latency_mean_ms": round(meta["latency_mean_ms"], 3),
                #"param_count_M": round(meta["param_count_M"], 3),
                "cost_per_1k_dkk": round(meta["cost_per_1k_dkk"], 3),
                "maintainability_score": round(meta["maintainability_score"], 3),
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

