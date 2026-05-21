from pathlib import Path
from datetime import datetime
import shutil
import hashlib
import pandas as pd
import cv2
from PIL import Image

BASE_DIR = Path("./data/external/mit_indoor")
IMAGES_DIR = BASE_DIR / "Images"
TRAIN_SPLIT = BASE_DIR / "TrainImages.txt"
TEST_SPLIT = BASE_DIR / "TestImages.txt"

TARGET_BASE = Path("./data/raw/mit_indoor_subset")
METADATA_OUT = Path("./data/processed/metadata/extracted/mit_indoor_subset_index.csv")

SELECTED_CLASSES = ["bathroom", "bedroom", "diningroom", "kitchen", "livingroom"]
LABEL_MAP = {cls_name: idx for idx, cls_name in enumerate(SELECTED_CLASSES)}

rows = []

def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def safe_exif_count(img):
    try:
        exif = img.getexif()
        return len(exif) if exif else 0
    except Exception:
        return 0

def extract_metadata(image_path: Path):
    pil_width = pil_height = channels = None
    image_mode = image_format = None
    exif_tag_count = 0

    try:
        with Image.open(image_path) as img:
            pil_width, pil_height = img.size
            image_mode = img.mode
            image_format = img.format
            exif_tag_count = safe_exif_count(img)
    except Exception:
        pass

    try:
        img_cv = cv2.imread(str(image_path))
        if img_cv is not None:
            if len(img_cv.shape) == 3:
                height, width, channels = img_cv.shape
            elif len(img_cv.shape) == 2:
                height, width = img_cv.shape
                channels = 1
            else:
                height = width = channels = None
        else:
            height = width = None
    except Exception:
        height = width = channels = None

    width = pil_width if pil_width is not None else width
    height = pil_height if pil_height is not None else height

    aspect_ratio = round(width / height, 4) if width and height else None
    pixel_count = width * height if width and height else None

    return {
        "width": width,
        "height": height,
        "channels": channels,
        "image_mode": image_mode,
        "image_format": image_format,
        "aspect_ratio": aspect_ratio,
        "pixel_count": pixel_count,
        "exif_tag_count": exif_tag_count
    }

def process_split(split_file: Path, split_name: str):
    counters = {class_name: 0 for class_name in SELECTED_CLASSES}

    with open(split_file, "r") as f:
        relative_paths = [line.strip() for line in f if line.strip()]

    for rel_path in relative_paths:
        class_name = rel_path.split("/")[0]
        if class_name not in SELECTED_CLASSES:
            continue

        src_path = IMAGES_DIR / rel_path
        if not src_path.exists():
            print(f"Missing file: {src_path}")
            continue

        counters[class_name] += 1
        file_ext = src_path.suffix.lower()
        new_filename = f"{class_name}_{counters[class_name]:06d}{file_ext}"

        target_dir = TARGET_BASE / split_name / class_name
        target_dir.mkdir(parents=True, exist_ok=True)

        dst_path = target_dir / new_filename
        shutil.copy2(src_path, dst_path)

        metadata = extract_metadata(dst_path)

        rows.append({
            "image_id": f"{split_name}_{class_name}_{counters[class_name]:06d}",
            "source_dataset": "mit_indoor",
            "split": split_name,
            "class_name": class_name,
            "label_id": LABEL_MAP[class_name],
            "original_relative_path": rel_path,
            "raw_path": str(dst_path),
            "original_filename": src_path.name,
            "new_filename": new_filename,
            "width": metadata["width"],
            "height": metadata["height"],
            "channels": metadata["channels"],
            "image_mode": metadata["image_mode"],
            "image_format": metadata["image_format"],
            "aspect_ratio": metadata["aspect_ratio"],
            "pixel_count": metadata["pixel_count"],
            "exif_tag_count": metadata["exif_tag_count"],
            "file_size_bytes": dst_path.stat().st_size,
            "md5": md5sum(dst_path),
            "ingestion_timestamp": datetime.utcnow().isoformat()
        })

def main():
    process_split(TRAIN_SPLIT, "train")
    process_split(TEST_SPLIT, "test")

    df = pd.DataFrame(rows)
    METADATA_OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(METADATA_OUT, index=False)

    print(f"Saved metadata to: {METADATA_OUT}")
    print("\nImage counts by split and class:")
    print(df.groupby(["split", "class_name"]).size().unstack(fill_value=0))

if __name__ == "__main__":
    main()


