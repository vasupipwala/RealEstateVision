from pathlib import Path
from datetime import datetime
import shutil
import sqlite3
import pandas as pd
import numpy as np
import cv2
from PIL import Image, UnidentifiedImageError
import imagehash

DB_PATH = "./db/realestatevision.db"
IMAGES_TABLE = "images"
VALIDATION_TABLE = "image_validation_results"

CLEANED_BASE = Path("./data/processed/cleaned")
REPORT_BASE = Path("./data/processed/validation_reports")
REPORT_BASE.mkdir(parents=True, exist_ok=True)
CLEANED_BASE.mkdir(parents=True, exist_ok=True)

BLUR_THRESHOLD = 100.0
DARK_THRESHOLD = 40.0
BRIGHT_THRESHOLD = 220.0
LOW_CONTRAST_THRESHOLD = 20.0
PHASH_DISTANCE_THRESHOLD = 5


def ensure_validation_table(conn):
    conn.execute(f"""
    CREATE TABLE IF NOT EXISTS {VALIDATION_TABLE} (
        image_id TEXT PRIMARY KEY,
        raw_path TEXT,
        cleaned_path TEXT,
        split TEXT,
        class_name TEXT,
        blur_score REAL,
        brightness_mean REAL,
        contrast_std REAL,
        phash TEXT,
        is_corrupt INTEGER,
        is_blurry INTEGER,
        too_dark INTEGER,
        too_bright INTEGER,
        low_contrast INTEGER,
        is_duplicate INTEGER,
        duplicate_of TEXT,
        duplicate_distance INTEGER,
        is_valid INTEGER,
        validation_timestamp TEXT,
        FOREIGN KEY(image_id) REFERENCES images(image_id)
    );
    """)

    existing_cols = {
        row[1] for row in conn.execute(f"PRAGMA table_info({VALIDATION_TABLE});").fetchall()
    }

    if "cleaned_path" not in existing_cols:
        conn.execute(f"ALTER TABLE {VALIDATION_TABLE} ADD COLUMN cleaned_path TEXT;")
    if "duplicate_distance" not in existing_cols:
        conn.execute(f"ALTER TABLE {VALIDATION_TABLE} ADD COLUMN duplicate_distance INTEGER;")

    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_validation_split ON {VALIDATION_TABLE}(split);")
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_validation_class_name ON {VALIDATION_TABLE}(class_name);")
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_validation_validity ON {VALIDATION_TABLE}(is_valid);")
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_validation_phash ON {VALIDATION_TABLE}(phash);")
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_validation_duplicate_of ON {VALIDATION_TABLE}(duplicate_of);")


def fetch_images_metadata(conn):
    query = f"""
    SELECT image_id, raw_path, split, class_name
    FROM {IMAGES_TABLE}
    ORDER BY split, class_name, image_id;
    """
    return pd.read_sql_query(query, conn)


def compute_quality_metrics(image_path: str):
    result = {
        "blur_score": None,
        "brightness_mean": None,
        "contrast_std": None,
        "phash": None,
        "is_corrupt": 0,
        "is_blurry": 0,
        "too_dark": 0,
        "too_bright": 0,
        "low_contrast": 0,
        "is_duplicate": 0,
        "duplicate_of": None,
        "duplicate_distance": None,
        "is_valid": 0
    }

    try:
        with Image.open(image_path) as pil_img:
            pil_img.verify()
    except (UnidentifiedImageError, OSError, ValueError):
        result["is_corrupt"] = 1
        return result

    try:
        with Image.open(image_path) as pil_img:
            pil_img = pil_img.convert("RGB")
            result["phash"] = imagehash.phash(pil_img)
    except Exception:
        result["is_corrupt"] = 1
        return result

    img_cv = cv2.imread(image_path)
    if img_cv is None:
        result["is_corrupt"] = 1
        return result

    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness_mean = float(np.mean(gray))
    contrast_std = float(np.std(gray))

    result["blur_score"] = blur_score
    result["brightness_mean"] = brightness_mean
    result["contrast_std"] = contrast_std
    result["is_blurry"] = int(blur_score < BLUR_THRESHOLD)
    result["too_dark"] = int(brightness_mean < DARK_THRESHOLD)
    result["too_bright"] = int(brightness_mean > BRIGHT_THRESHOLD)
    result["low_contrast"] = int(contrast_std < LOW_CONTRAST_THRESHOLD)

    return result


def find_near_duplicate(current_hash, seen_hashes):
    """
    seen_hashes: list of tuples (image_id, imagehash_object)
    Returns (duplicate_of, duplicate_distance) or (None, None)
    """
    best_match_id = None
    best_distance = None

    for seen_image_id, seen_hash in seen_hashes:
        distance = current_hash - seen_hash
        if distance <= PHASH_DISTANCE_THRESHOLD:
            if best_distance is None or distance < best_distance:
                best_match_id = seen_image_id
                best_distance = distance

    return best_match_id, best_distance


def copy_valid_image(src_path: Path, split: str, class_name: str):
    target_dir = CLEANED_BASE / split / class_name
    target_dir.mkdir(parents=True, exist_ok=True)
    dst_path = target_dir / src_path.name
    shutil.copy2(src_path, dst_path)
    return str(dst_path)


def main():
    conn = sqlite3.connect(DB_PATH)
    ensure_validation_table(conn)

    images_df = fetch_images_metadata(conn)
    results = []

    seen_hashes = []

    for _, row in images_df.iterrows():
        image_id = row["image_id"]
        raw_path = row["raw_path"]
        split = row["split"]
        class_name = row["class_name"]

        metrics = compute_quality_metrics(raw_path)

        current_hash = metrics["phash"]

        if current_hash is not None:
            duplicate_of, duplicate_distance = find_near_duplicate(current_hash, seen_hashes)

            if duplicate_of is not None:
                metrics["is_duplicate"] = 1
                metrics["duplicate_of"] = duplicate_of
                metrics["duplicate_distance"] = int(duplicate_distance)

            seen_hashes.append((image_id, current_hash))

        is_valid = int(
            metrics["is_corrupt"] == 0 and
            metrics["is_blurry"] == 0 and
            metrics["too_dark"] == 0 and
            metrics["too_bright"] == 0 and
            metrics["low_contrast"] == 0 and
            metrics["is_duplicate"] == 0
        )
        metrics["is_valid"] = is_valid

        cleaned_path = None
        if is_valid:
            cleaned_path = copy_valid_image(Path(raw_path), split, class_name)

        results.append({
            "image_id": image_id,
            "raw_path": raw_path,
            "cleaned_path": cleaned_path,
            "split": split,
            "class_name": class_name,
            "blur_score": metrics["blur_score"],
            "brightness_mean": metrics["brightness_mean"],
            "contrast_std": metrics["contrast_std"],
            "phash": str(metrics["phash"]) if metrics["phash"] is not None else None,
            "is_corrupt": metrics["is_corrupt"],
            "is_blurry": metrics["is_blurry"],
            "too_dark": metrics["too_dark"],
            "too_bright": metrics["too_bright"],
            "low_contrast": metrics["low_contrast"],
            "is_duplicate": metrics["is_duplicate"],
            "duplicate_of": metrics["duplicate_of"],
            "duplicate_distance": metrics["duplicate_distance"],
            "is_valid": metrics["is_valid"],
            "validation_timestamp": datetime.utcnow().isoformat()
        })

    results_df = pd.DataFrame(results)

    results_df.to_csv(REPORT_BASE / "image_validation_results.csv", index=False)

    summary_df = pd.DataFrame([{
        "total_images": len(results_df),
        "valid_images": int(results_df["is_valid"].sum()),
        "corrupt_images": int(results_df["is_corrupt"].sum()),
        "blurry_images": int(results_df["is_blurry"].sum()),
        "too_dark_images": int(results_df["too_dark"].sum()),
        "too_bright_images": int(results_df["too_bright"].sum()),
        "low_contrast_images": int(results_df["low_contrast"].sum()),
        "duplicate_images": int(results_df["is_duplicate"].sum()),
        "valid_ratio": round(results_df["is_valid"].mean(), 4)
    }])
    summary_df.to_csv(REPORT_BASE / "validation_summary.csv", index=False)

    split_class_summary = (
        results_df.groupby(["split", "class_name"])
        .agg(
            total_images=("image_id", "count"),
            valid_images=("is_valid", "sum"),
            blurry_images=("is_blurry", "sum"),
            dark_images=("too_dark", "sum"),
            bright_images=("too_bright", "sum"),
            low_contrast_images=("low_contrast", "sum"),
            duplicate_images=("is_duplicate", "sum"),
            corrupt_images=("is_corrupt", "sum")
        )
        .reset_index()
    )
    split_class_summary.to_csv(REPORT_BASE / "validation_by_split_class.csv", index=False)

    duplicate_pairs = results_df[results_df["is_duplicate"] == 1][[
        "image_id", "duplicate_of", "duplicate_distance", "raw_path", "class_name", "split"
    ]]
    duplicate_pairs.to_csv(REPORT_BASE / "near_duplicate_pairs.csv", index=False)

    conn.execute(f"DELETE FROM {VALIDATION_TABLE};")
    results_df.to_sql(VALIDATION_TABLE, conn, if_exists="append", index=False)
    conn.commit()
    conn.close()

    print("Validation complete.")
    print(f"Row-level report: {REPORT_BASE / 'image_validation_results.csv'}")
    print(f"Summary report: {REPORT_BASE / 'validation_summary.csv'}")
    print(f"Split/class report: {REPORT_BASE / 'validation_by_split_class.csv'}")
    print(f"Near-duplicate report: {REPORT_BASE / 'near_duplicate_pairs.csv'}")
    print(f"Cleaned dataset: {CLEANED_BASE}")
    print(f"SQL table updated: {VALIDATION_TABLE}")


if __name__ == "__main__":
    main()
