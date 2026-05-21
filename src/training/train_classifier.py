"""
Room Type Classification — ML Pipeline
================================================
Models   : MobileNetV3-Small, EfficientNet-B0, ResNet-18
Tracking : MLflow (runs linked to dataset_versions.version_tag)
Artifacts: models/  checkpoints/  data/processed/evaluation_reports/

Usage:
    python src/training/train_classifier.py \
        --db-path db/realestatevision.db \
        --version-tag cleaned_v1 \
        --epochs 15 \
        --batch-size 32 \
        --lr 1e-3 \
        --seed 42
"""

import argparse
import json
import platform
import sqlite3
import time
from datetime import datetime
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    top_k_accuracy_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import (
    EfficientNet_B0_Weights,
    MobileNet_V3_Small_Weights,
    ResNet18_Weights,
    efficientnet_b0,
    mobilenet_v3_small,
    resnet18,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
DB_PATH = "./db/realestatevision.db"
MODELS_DIR = Path("models")
CHECKPOINTS_DIR = Path("checkpoints")
EVAL_REPORTS_DIR = Path("./data/processed/evaluation_reports")
MLFLOW_TRACKING_URI = "mlruns"
MLFLOW_EXPERIMENT_NAME = "room_type_classification"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
EVAL_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
CLASSES = ["bathroom", "bedroom", "diningroom", "kitchen", "livingroom"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

IMG_SIZE = 224
NUM_WORKERS = 0
SEED = 42

# Cost baseline:
# User-observed local benchmark = 0.5 DKK for 500 images on laptop
# => 1.0 DKK for 1000 images at baseline throughput
BASELINE_COST_PER_1K_DKK = 1.0
BASELINE_COST_REFERENCE = "0.5_dkk_per_500_images_local_laptop"
SCORING_VERSION = "v2"

# Maintainability rubric weights
MAINTAINABILITY_WEIGHTS = {
    "param_count": 0.4,
    "trainable_params": 0.4,
    "dependency_count": 0.2,
}

# Simple explicit dependency counts for inference/training complexity.
# These are now treated as transparent assumptions, not hidden qualitative labels.
MODEL_DEPENDENCY_COUNT = {
    "mobilenet_v3_small": 1,
    "efficientnet_b0": 1,
    "resnet18": 1,
}


# ── Device ────────────────────────────────────────────────────────────────────
def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── Data ──────────────────────────────────────────────────────────────────────
def load_manifest(db_path: str, version_tag: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load valid train/test image paths from SQLite using image_validation_results."""
    conn = sqlite3.connect(db_path)

    df = pd.read_sql_query(
        """
        SELECT image_id, cleaned_path, split, class_name
        FROM image_validation_results
        WHERE is_valid = 1 AND cleaned_path IS NOT NULL
        """,
        conn,
    )
    conn.close()

    df = df[df["class_name"].isin(CLASSES)].reset_index(drop=True)
    df["label"] = df["class_name"].map(CLASS_TO_IDX)

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    print(f"Train: {len(train_df)} | Test: {len(test_df)}")
    print("Class distribution (train):")
    print(train_df["class_name"].value_counts().to_string())
    return train_df, test_df


class RoomDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.paths = df["cleaned_path"].tolist()
        self.labels = df["label"].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def get_transforms(split: str):
    if split == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize(int(IMG_SIZE * 1.14)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


# ── Models ────────────────────────────────────────────────────────────────────
def build_model(name: str, num_classes: int) -> nn.Module:
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


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ── Training ──────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        if scaler is not None:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                logits = model(imgs)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels, all_probs = 0.0, [], [], []
    total = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)

        total_loss += loss.item() * imgs.size(0)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
        all_probs.append(probs)
        total += imgs.size(0)

    all_probs = np.vstack(all_probs)
    return (
        total_loss / total,
        all_labels,
        all_preds,
        all_probs,
    )


# ── Inference timing ──────────────────────────────────────────────────────────
@torch.no_grad()
def measure_inference_latency(model, device, n_warmup=10, n_runs=100) -> dict:
    """Measure mean/p95/p99 latency for single-image inference (ms)."""
    model.eval()
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)
    latencies = []

    for _ in range(n_warmup):
        _ = model(dummy)

    for _ in range(n_runs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()

        t0 = time.perf_counter()
        _ = model(dummy)

        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()

        latencies.append((time.perf_counter() - t0) * 1000)

    return {
        "latency_mean_ms": round(float(np.mean(latencies)), 3),
        "latency_p50_ms": round(float(np.percentile(latencies, 50)), 3),
        "latency_p95_ms": round(float(np.percentile(latencies, 95)), 3),
        "latency_p99_ms": round(float(np.percentile(latencies, 99)), 3),
        "throughput_img_per_sec": round(1000.0 / float(np.mean(latencies)), 1),
    }


@torch.no_grad()
def measure_batch_throughput(model, loader, device) -> dict:
    """Measure throughput on real test batches."""
    model.eval()
    total_imgs = 0
    t0 = time.perf_counter()

    for imgs, _ in loader:
        imgs = imgs.to(device)
        _ = model(imgs)
        total_imgs += imgs.size(0)

    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()

    elapsed = time.perf_counter() - t0
    return {
        "batch_throughput_img_per_sec": round(total_imgs / elapsed, 1),
        "total_inference_sec": round(elapsed, 4),
    }


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_quality_metrics(labels, preds, probs, class_names) -> dict:
    acc = accuracy_score(labels, preds)
    f1_w = f1_score(labels, preds, average="weighted", zero_division=0)
    f1_m = f1_score(labels, preds, average="macro", zero_division=0)
    mcc = matthews_corrcoef(labels, preds)
    kappa = cohen_kappa_score(labels, preds)
    top2 = top_k_accuracy_score(labels, probs, k=2)

    report = classification_report(
        labels, preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(labels, preds).tolist()

    return {
        "accuracy": round(acc, 4),
        "f1_weighted": round(f1_w, 4),
        "f1_macro": round(f1_m, 4),
        "mcc": round(mcc, 4),
        "cohen_kappa": round(kappa, 4),
        "top_2_accuracy": round(top2, 4),
        "per_class_report": report,
        "confusion_matrix": cm,
    }


def compute_cost_metrics(
    throughput_img_per_sec: float,
    n_images: int,
    baseline_cost_per_1k_dkk: float = BASELINE_COST_PER_1K_DKK,
    baseline_throughput_img_per_sec: float = 100.0,
) -> dict:
    """
    Cost model:
    - Shared base cost for all models: 1.0 DKK / 1000 images at baseline throughput.
    - Slower models cost more because they take longer to process the same workload.
    - This is a local-hardware relative cost estimate, not a cloud billing number.
    """
    if throughput_img_per_sec <= 0:
        raise ValueError("throughput_img_per_sec must be > 0")

    relative_compute_multiplier = baseline_throughput_img_per_sec / throughput_img_per_sec
    cost_per_1k = baseline_cost_per_1k_dkk * relative_compute_multiplier

    return {
        "cost_per_1k_images_dkk": round(cost_per_1k, 4),
        "cost_for_test_set_dkk": round(cost_per_1k * n_images / 1000, 5),
        "cost_scale_10k_dkk": round(cost_per_1k * 10, 4),
        "cost_scale_100k_dkk": round(cost_per_1k * 100, 4),
        "cost_scale_1M_dkk": round(cost_per_1k * 1000, 4),
        "relative_compute_multiplier": round(relative_compute_multiplier, 4),
        "baseline_cost_per_1k_dkk": round(baseline_cost_per_1k_dkk, 4),
        "baseline_throughput_img_per_sec": round(baseline_throughput_img_per_sec, 4),
    }


def _score_smaller_is_better(value: float, thresholds: tuple[float, float, float, float]) -> int:
    """
    Convert a metric into a 1-5 score where lower is better.
    thresholds are upper bounds for scores 5, 4, 3, 2; above last bound gets 1.
    """
    if value <= thresholds[0]:
        return 5
    if value <= thresholds[1]:
        return 4
    if value <= thresholds[2]:
        return 3
    if value <= thresholds[3]:
        return 2
    return 1


def compute_maintainability_metrics(model_name: str, model: nn.Module) -> dict:
    """
    Maintainability rubric:
    - Smaller models are easier to store, inspect, and deploy.
    - Fewer trainable parameters reduce retraining/tuning burden.
    - Fewer extra dependencies reduce operational complexity.
    """
    total_params = count_total_params(model)
    trainable_params = count_trainable_params(model)

    param_count_m = total_params / 1_000_000
    trainable_params_m = trainable_params / 1_000_000
    dependency_count = MODEL_DEPENDENCY_COUNT.get(model_name, 1)

    param_score = _score_smaller_is_better(param_count_m, (3.0, 6.0, 12.0, 20.0))
    trainable_score = _score_smaller_is_better(trainable_params_m, (3.0, 6.0, 12.0, 20.0))
    dependency_score = _score_smaller_is_better(dependency_count, (1, 2, 3, 4))

    weighted_score = (
        MAINTAINABILITY_WEIGHTS["param_count"] * param_score +
        MAINTAINABILITY_WEIGHTS["trainable_params"] * trainable_score +
        MAINTAINABILITY_WEIGHTS["dependency_count"] * dependency_score
    )

    return {
        "maintainability_score": round(weighted_score, 3),
        "param_count_M": round(param_count_m, 3),
        "trainable_params_M": round(trainable_params_m, 3),
        "trainable_params": int(trainable_params),
        "dependency_count": int(dependency_count),
        "maint_param_score_1_5": int(param_score),
        "maint_trainable_score_1_5": int(trainable_score),
        "maint_dependency_score_1_5": int(dependency_score),
    }


# ── Dataset version lookup ────────────────────────────────────────────────────
def get_version_record(db_path: str, version_tag: str) -> dict:
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT * FROM dataset_versions WHERE version_tag = ?",
        (version_tag,)
    ).fetchone()
    cols = [d[0] for d in conn.execute(
        "SELECT * FROM dataset_versions LIMIT 0"
    ).description] if row else []
    conn.close()
    if row:
        return dict(zip(cols, row))
    return {"version_tag": version_tag, "git_commit": "unknown"}


# ── Main training loop ────────────────────────────────────────────────────────
def train_model(
    model_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    version_record: dict,
    args,
    device: torch.device,
    experiment_id: str,
):
    print(f"\n{'='*60}")
    print(f"  Training: {model_name}")
    print(f"  Device:   {device}")
    print(f"{'='*60}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_ds = RoomDataset(train_df, get_transforms("train"))
    test_ds = RoomDataset(test_df, get_transforms("test"))

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=False
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False
    )

    model = build_model(model_name, num_classes=len(CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5
    )
    scaler = None

    run_name = f"{model_name}_{version_record['version_tag']}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
        total_params = count_total_params(model)
        trainable_params = count_trainable_params(model)

        # Dataset/version params
        mlflow.log_param("experiment_name", MLFLOW_EXPERIMENT_NAME)
        mlflow.log_param("scoring_version", SCORING_VERSION)
        mlflow.log_param("dataset_version_tag", version_record["version_tag"])
        mlflow.log_param("git_commit", version_record.get("git_commit", "n/a"))
        mlflow.log_param("raw_dvc_md5", version_record.get("raw_dvc_md5", "n/a"))
        mlflow.log_param("cleaned_dvc_md5", version_record.get("cleaned_dvc_md5", "n/a"))
        mlflow.log_param("raw_image_count", version_record.get("raw_image_count", -1))
        mlflow.log_param("cleaned_image_count", version_record.get("cleaned_image_count", -1))

        # Training/system params
        mlflow.log_param("model", model_name)
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("lr", args.lr)
        mlflow.log_param("seed", args.seed)
        mlflow.log_param("img_size", IMG_SIZE)
        mlflow.log_param("device", str(device))
        mlflow.log_param("platform", platform.platform())
        mlflow.log_param("num_classes", len(CLASSES))
        mlflow.log_param("total_params", total_params)
        mlflow.log_param("trainable_params", trainable_params)

        # Formula assumptions
        mlflow.log_param("cost_baseline_reference", BASELINE_COST_REFERENCE)
        mlflow.log_param("cost_formula", "cost_per_1k_images_dkk = baseline_cost_per_1k_dkk * (baseline_throughput / throughput_img_per_sec)")
        mlflow.log_param("maintainability_formula", "0.4*param_score + 0.4*trainable_score + 0.2*dependency_score")
        mlflow.log_param("maintainability_param_thresholds_M", "3,6,12,20")
        mlflow.log_param("maintainability_dependency_thresholds", "1,2,3,4")
        mlflow.log_param("dependency_count_assumption", MODEL_DEPENDENCY_COUNT.get(model_name, 1))

        best_acc = 0.0
        best_epoch = 0
        history = []

        for epoch in range(1, args.epochs + 1):
            t0 = time.perf_counter()
            tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
            val_loss, labels, preds, probs = evaluate(model, test_loader, criterion, device)
            val_acc = accuracy_score(labels, preds)
            val_f1 = f1_score(labels, preds, average="weighted", zero_division=0)
            elapsed = time.perf_counter() - t0

            scheduler.step()

            mlflow.log_metrics({
                "train_loss": round(tr_loss, 5),
                "train_acc": round(tr_acc, 4),
                "val_loss": round(val_loss, 5),
                "val_acc": round(val_acc, 4),
                "val_f1_weighted": round(val_f1, 4),
                "epoch_time_sec": round(elapsed, 2),
                "lr": round(scheduler.get_last_lr()[0], 8),
            }, step=epoch)

            history.append({
                "epoch": epoch,
                "train_loss": round(tr_loss, 5),
                "train_acc": round(tr_acc, 4),
                "val_loss": round(val_loss, 5),
                "val_acc": round(val_acc, 4),
                "val_f1_weighted": round(val_f1, 4),
            })

            print(
                f"  Epoch {epoch:02d}/{args.epochs} | "
                f"train_loss={tr_loss:.4f} acc={tr_acc:.4f} | "
                f"val_loss={val_loss:.4f} acc={val_acc:.4f} f1={val_f1:.4f} | "
                f"{elapsed:.1f}s"
            )

            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch
                ckpt_path = CHECKPOINTS_DIR / f"{model_name}_best.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "val_acc": val_acc,
                    "val_f1": val_f1,
                    "version_tag": version_record["version_tag"],
                }, ckpt_path)

        print(f"  Best val_acc={best_acc:.4f} at epoch {best_epoch}")

        val_loss, labels, preds, probs = evaluate(model, test_loader, criterion, device)

        quality_metrics = compute_quality_metrics(labels, preds, probs, CLASSES)
        perf_metrics = measure_inference_latency(model, device)
        batch_perf = measure_batch_throughput(model, test_loader, device)
        cost_metrics = compute_cost_metrics(
            throughput_img_per_sec=perf_metrics["throughput_img_per_sec"],
            n_images=len(test_df),
            baseline_cost_per_1k_dkk=BASELINE_COST_PER_1K_DKK,
            baseline_throughput_img_per_sec=100,
        )
        maint_metrics = compute_maintainability_metrics(model_name, model)

        mlflow.log_metrics({
            "final_accuracy": quality_metrics["accuracy"],
            "final_f1_weighted": quality_metrics["f1_weighted"],
            "final_f1_macro": quality_metrics["f1_macro"],
            "final_mcc": quality_metrics["mcc"],
            "final_cohen_kappa": quality_metrics["cohen_kappa"],
            "final_top2_accuracy": quality_metrics["top_2_accuracy"],
            "best_val_acc": round(best_acc, 4),
            "best_epoch": best_epoch,
        })

        mlflow.log_metrics({**perf_metrics, **batch_perf})
        mlflow.log_metrics(cost_metrics)
        mlflow.log_metrics(maint_metrics)

        final_model_path = MODELS_DIR / f"{model_name}_{version_record['version_tag']}.pt"
        torch.save(model.state_dict(), final_model_path)
        mlflow.log_artifact(str(final_model_path), artifact_path="model_weights")
        mlflow.pytorch.log_model(model, name="model")

        report = {
            "model": model_name,
            "version_tag": version_record["version_tag"],
            "run_id": run.info.run_id,
            "trained_at": datetime.utcnow().isoformat(),
            "scoring_version": SCORING_VERSION,
            "quality": quality_metrics,
            "performance": {**perf_metrics, **batch_perf},
            "cost": cost_metrics,
            "maintainability": maint_metrics,
            "training_history": history,
            "assumptions": {
                "cost_baseline_reference": BASELINE_COST_REFERENCE,
                "cost_formula": "cost_per_1k_images_dkk = baseline_cost_per_1k_dkk * (baseline_throughput / throughput_img_per_sec)",
                "maintainability_formula": "0.4*param_score + 0.4*trainable_score + 0.2*dependency_score",
            },
        }

        report_path = EVAL_REPORTS_DIR / f"{model_name}_{version_record['version_tag']}_report.json"
        report_path.write_text(json.dumps(report, indent=2))
        mlflow.log_artifact(str(report_path), artifact_path="evaluation_reports")

        per_class_df = pd.DataFrame(quality_metrics["per_class_report"]).T
        per_class_csv = EVAL_REPORTS_DIR / f"{model_name}_{version_record['version_tag']}_per_class.csv"
        per_class_df.to_csv(per_class_csv)
        mlflow.log_artifact(str(per_class_csv), artifact_path="evaluation_reports")

        hist_csv = EVAL_REPORTS_DIR / f"{model_name}_{version_record['version_tag']}_history.csv"
        pd.DataFrame(history).to_csv(hist_csv, index=False)
        mlflow.log_artifact(str(hist_csv), artifact_path="evaluation_reports")

        cm_path = EVAL_REPORTS_DIR / f"{model_name}_{version_record['version_tag']}_confusion_matrix.json"
        cm_path.write_text(json.dumps({
            "classes": CLASSES,
            "matrix": quality_metrics["confusion_matrix"]
        }, indent=2))
        mlflow.log_artifact(str(cm_path), artifact_path="evaluation_reports")

        print(
            f"\n  Quality  | accuracy={quality_metrics['accuracy']:.4f}  "
            f"f1_w={quality_metrics['f1_weighted']:.4f}  "
            f"mcc={quality_metrics['mcc']:.4f}  "
            f"kappa={quality_metrics['cohen_kappa']:.4f}"
        )
        print(
            f"  Perf     | latency={perf_metrics['latency_mean_ms']:.1f}ms  "
            f"throughput={perf_metrics['throughput_img_per_sec']:.0f}img/s"
        )
        print(
            f"  Cost     | {cost_metrics['cost_per_1k_images_dkk']:.2f} kr./1k imgs  "
            f"{cost_metrics['cost_scale_1M_dkk']:.2f} kr./1M imgs"
        )
        print(
            f"  Maintain | score={maint_metrics['maintainability_score']:.2f}/5  "
            f"params={maint_metrics['param_count_M']:.3f}M"
        )
        print(f"  Saved    | {final_model_path}  |  {report_path}")

    return report


# ── Comparison table ──────────────────────────────────────────────────────────
def save_comparison_table(reports: list[dict], version_tag: str):
    rows = []
    for r in reports:
        rows.append({
            "model": r["model"],
            "version_tag": r["version_tag"],
            "accuracy": r["quality"]["accuracy"],
            "f1_weighted": r["quality"]["f1_weighted"],
            "f1_macro": r["quality"]["f1_macro"],
            "mcc": r["quality"]["mcc"],
            "cohen_kappa": r["quality"]["cohen_kappa"],
            "top_2_accuracy": r["quality"]["top_2_accuracy"],
            "latency_mean_ms": r["performance"]["latency_mean_ms"],
            "latency_p95_ms": r["performance"]["latency_p95_ms"],
            "throughput_img_per_sec": r["performance"]["throughput_img_per_sec"],
            "batch_throughput_img_s": r["performance"]["batch_throughput_img_per_sec"],
            "total_inference_sec": r["performance"]["total_inference_sec"],
            "cost_per_1k_dkk": r["cost"]["cost_per_1k_images_dkk"],
            "cost_scale_10k_dkk": r["cost"]["cost_scale_10k_dkk"],
            "cost_scale_100k_dkk": r["cost"]["cost_scale_100k_dkk"],
            "cost_scale_1M_dkk": r["cost"]["cost_scale_1M_dkk"],
            "relative_compute_multiplier": r["cost"]["relative_compute_multiplier"],
            "maintainability_score": r["maintainability"]["maintainability_score"],
            "dependency_count": r["maintainability"]["dependency_count"],
            "param_count_M": r["maintainability"]["param_count_M"],
            "trainable_params_M": r["maintainability"]["trainable_params_M"],
            "trainable_params": r["maintainability"]["trainable_params"],
        })

    df = pd.DataFrame(rows)
    out = EVAL_REPORTS_DIR / f"model_comparison_{version_tag}.csv"
    df.to_csv(out, index=False)
    print(f"\n✅ Comparison table saved: {out}")
    print(df[[
        "model", "accuracy", "f1_weighted", "latency_mean_ms",
        "throughput_img_per_sec", "cost_per_1k_dkk", "maintainability_score"
    ]].to_string(index=False))


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", default=DB_PATH)
    parser.add_argument("--version-tag", required=True)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["mobilenet_v3_small", "efficientnet_b0", "resnet18"]
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = get_device()
    print(f"Device: {device}")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    experiment = mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    train_df, test_df = load_manifest(args.db_path, args.version_tag)
    version_record = get_version_record(args.db_path, args.version_tag)
    print(
        f"\nDataset version: {version_record['version_tag']} | "
        f"git={version_record.get('git_commit','n/a')[:10]}"
    )

    all_reports = []
    for model_name in args.models:
        report = train_model(
            model_name=model_name,
            train_df=train_df,
            test_df=test_df,
            version_record=version_record,
            args=args,
            device=device,
            experiment_id=experiment.experiment_id,
        )
        all_reports.append(report)

    save_comparison_table(all_reports, args.version_tag)

    print("\n✅ train_classifier.py script complete.")
    print(f"   MLflow experiment: {MLFLOW_EXPERIMENT_NAME}")
    print(f"   MLflow UI: mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI}")


if __name__ == "__main__":
    main()
