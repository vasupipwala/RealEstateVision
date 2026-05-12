from pathlib import Path
import pandas as pd
import shutil
from PIL import Image
import hashlib
from datetime import datetime

BASE = Path("./data/external/mit_indoor/")
IMAGES_DIR = BASE / "Images"
TRAIN_FILE = BASE / "TrainImages.txt"
TEST_FILE = BASE / "TestImages.txt"

TARGET_BASE = Path("./data/raw/mit_indoor_subset")
METADATA_OUT = Path("./data/processed/metadata/extracted/mit_indoor_subset_index.csv")

SELECTED_CLASSES = {
    "bathroom",
    "bedroom",
    "dining_room",
    "kitchen",
    "livingroom"
}

LABEL_MAP = {
    "bathroom": 0,
    "bedroom": 1,
    "dining_room": 2,
    "kitchen": 3,
    "livingroom": 4
}

rows = []

def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def process_split(split_file: Path, split_name: str):
    with open(split_file, "r") as f:
        rel_paths = [line.strip() for line in f if line.strip()]

    for rel_path in rel_paths:
        class_name = rel_path.split("/")[0]
        if class_name not in SELECTED_CLASSES:
            continue

        src_path = IMAGES_DIR / rel_path
        if not src_path.exists():
            print(f"Missing file: {src_path}")
            continue

        target_dir = TARGET_BASE / split_name / class_name
        target_dir.mkdir(parents=True, exist_ok=True)
        dst_path = target_dir / src_path.name
        shutil.copy2(src_path, dst_path)

        try:
            with Image.open(dst_path) as img:
                width, height = img.size
                mode = img.mode
                fmt = img.format
                channels = len(img.getbands()) if hasattr(img, "getbands") else None
        except Exception:
            width, height, mode, fmt, channels = None, None, None, None, None

        aspect_ratio = round(width / height, 4) if width and height else None
        pixel_count = width * height if width and height else None
        extension = dst_path.suffix.lower()
        is_rgb = mode == "RGB" if mode else None

        rows.append({
            "source_dataset": "mit_indoor",
            "split": split_name,
            "class_name": class_name,
            "original_relative_path": rel_path,
            "raw_path": str(dst_path),
            "filename": dst_path.name,
            "width": width,
            "height": height,
            "channels": channels,
            "image_mode": mode,
            "image_format": fmt,
            "file_size_bytes": dst_path.stat().st_size,
            "aspect_ratio": aspect_ratio,
            "pixel_count": pixel_count,
            "is_rgb": is_rgb,
            "extension": extension,
            "label_id": LABEL_MAP[class_name],
            "md5": md5sum(dst_path),
            "ingestion_timestamp": datetime.utcnow().isoformat()
        })

process_split(TRAIN_FILE, "train")
process_split(TEST_FILE, "test")

df = pd.DataFrame(rows)
METADATA_OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(METADATA_OUT, index=False)

print(f"Saved metadata to {METADATA_OUT}")
print(df.groupby(['split', 'class_name']).size())

