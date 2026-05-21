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

TENENGRAD_PERCENTILE = 5
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
        tenengrad_score REAL,
        laplacian_var REAL,
        brightness_mean REAL,
        contrast_std REAL,
        phash TEXT,
        is_corrupt INTEGER,
        is_blurry INTEGER,
        too_dark INTEGER,
        too_bright INTEGER,
        low_contrast INTEGER,
        is_duplicate INTEGER,
        canonical_image_id TEXT,
        duplicate_distance INTEGER,
        cross_split_duplicate INTEGER,
        cross_class_duplicate INTEGER,
        manual_review_required INTEGER,
        drop_reason TEXT,
        is_valid INTEGER,
        validation_timestamp TEXT,
        FOREIGN KEY(image_id) REFERENCES images(image_id)
    );
    """)

    existing_cols = {row[1] for row in conn.execute(f"PRAGMA table_info({VALIDATION_TABLE});").fetchall()}

    extra_columns = {
        "cleaned_path": "TEXT",
        "tenengrad_score": "REAL",
        "laplacian_var": "REAL",
        "canonical_image_id": "TEXT",
        "duplicate_distance": "INTEGER",
        "cross_split_duplicate": "INTEGER",
        "cross_class_duplicate": "INTEGER",
        "manual_review_required": "INTEGER",
        "drop_reason": "TEXT",
    }

    for col_name, col_type in extra_columns.items():
        if col_name not in existing_cols:
            conn.execute(f"ALTER TABLE {VALIDATION_TABLE} ADD COLUMN {col_name} {col_type};")

    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_validation_split ON {VALIDATION_TABLE}(split);")
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_validation_class_name ON {VALIDATION_TABLE}(class_name);")
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_validation_validity ON {VALIDATION_TABLE}(is_valid);")
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_validation_phash ON {VALIDATION_TABLE}(phash);")
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_validation_canonical_image_id ON {VALIDATION_TABLE}(canonical_image_id);")


def fetch_images_metadata(conn):
    query = f"""
    SELECT image_id, raw_path, split, class_name
    FROM {IMAGES_TABLE}
    ORDER BY split, class_name, image_id;
    """
    return pd.read_sql_query(query, conn)


def compute_tenengrad(gray_image):
    gx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag_sq = gx ** 2 + gy ** 2
    return float(np.mean(grad_mag_sq))


def compute_quality_metrics(image_path: str):
    result = {
        "tenengrad_score": None,
        "laplacian_var": None,
        "brightness_mean": None,
        "contrast_std": None,
        "phash_obj": None,
        "phash": None,
        "is_corrupt": 0,
        "is_blurry": 0,
        "too_dark": 0,
        "too_bright": 0,
        "low_contrast": 0,
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
            phash_obj = imagehash.phash(pil_img)
            result["phash_obj"] = phash_obj
            result["phash"] = str(phash_obj)
    except Exception:
        result["is_corrupt"] = 1
        return result

    img_cv = cv2.imread(image_path)
    if img_cv is None:
        result["is_corrupt"] = 1
        return result

    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    result["tenengrad_score"] = compute_tenengrad(gray)
    result["laplacian_var"] = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    result["brightness_mean"] = float(np.mean(gray))
    result["contrast_std"] = float(np.std(gray))
    result["too_dark"] = int(result["brightness_mean"] < DARK_THRESHOLD)
    result["too_bright"] = int(result["brightness_mean"] > BRIGHT_THRESHOLD)
    result["low_contrast"] = int(result["contrast_std"] < LOW_CONTRAST_THRESHOLD)

    return result


def quality_score_for_retention(row):
    if row["is_corrupt"] == 1:
        return -1e12
    return (
        float(row["tenengrad_score"] or 0.0)
        + 0.01 * float(row["laplacian_var"] or 0.0)
        + 0.001 * float(row["contrast_std"] or 0.0)
    )


def split_priority(split_name):
    return 1 if split_name == "train" else 0


def choose_canonical(group_df):
    candidate_df = group_df.copy()
    candidate_df["quality_rank_score"] = candidate_df.apply(quality_score_for_retention, axis=1)
    candidate_df["split_priority"] = candidate_df["split"].apply(split_priority)

    candidate_df = candidate_df.sort_values(
        by=["split_priority", "quality_rank_score", "image_id"],
        ascending=[False, False, True]
    )
    return candidate_df.iloc[0]["image_id"]


def detect_duplicate_groups(df):
    records = df.to_dict("records")
    n = len(records)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        if records[i]["phash_obj"] is None:
            continue
        for j in range(i + 1, n):
            if records[j]["phash_obj"] is None:
                continue
            dist = records[i]["phash_obj"] - records[j]["phash_obj"]
            if dist <= PHASH_DISTANCE_THRESHOLD:
                union(i, j)

    groups = {}
    for idx in range(n):
        root = find(idx)
        groups.setdefault(root, []).append(idx)

    return [group for group in groups.values() if len(group) > 1]


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

    rows = []
    for _, row in images_df.iterrows():
        metrics = compute_quality_metrics(row["raw_path"])
        rows.append({
            "image_id": row["image_id"],
            "raw_path": row["raw_path"],
            "split": row["split"],
            "class_name": row["class_name"],
            "tenengrad_score": metrics["tenengrad_score"],
            "laplacian_var": metrics["laplacian_var"],
            "brightness_mean": metrics["brightness_mean"],
            "contrast_std": metrics["contrast_std"],
            "phash_obj": metrics["phash_obj"],
            "phash": metrics["phash"],
            "is_corrupt": metrics["is_corrupt"],
            "is_blurry": 0,
            "too_dark": metrics["too_dark"],
            "too_bright": metrics["too_bright"],
            "low_contrast": metrics["low_contrast"],
            "is_duplicate": 0,
            "canonical_image_id": None,
            "duplicate_distance": None,
            "cross_split_duplicate": 0,
            "cross_class_duplicate": 0,
            "manual_review_required": 0,
            "drop_reason": None,
            "cleaned_path": None,
            "is_valid": 0,
            "validation_timestamp": datetime.utcnow().isoformat()
        })

    results_df = pd.DataFrame(rows)

    train_scores = results_df[
        (results_df["split"] == "train") &
        (results_df["is_corrupt"] == 0) &
        (results_df["tenengrad_score"].notna())
    ]["tenengrad_score"]

    tenengrad_threshold = float(np.percentile(train_scores, TENENGRAD_PERCENTILE)) if len(train_scores) > 0 else 0.0

    results_df["is_blurry"] = (
        (results_df["is_corrupt"] == 0) &
        (results_df["tenengrad_score"].fillna(-1) <= tenengrad_threshold)
    ).astype(int)

    duplicate_groups = detect_duplicate_groups(results_df)

    for group_indices in duplicate_groups:
        group_df = results_df.iloc[group_indices].copy()
        canonical_image_id = choose_canonical(group_df)
        canonical_row = group_df[group_df["image_id"] == canonical_image_id].iloc[0]

        same_split = len(group_df["split"].unique()) == 1
        same_class = len(group_df["class_name"].unique()) == 1

        for idx in group_indices:
            row = results_df.loc[idx]
            if row["image_id"] == canonical_image_id:
                results_df.at[idx, "canonical_image_id"] = canonical_image_id
                continue

            results_df.at[idx, "is_duplicate"] = 1
            results_df.at[idx, "canonical_image_id"] = canonical_image_id

            if row["phash_obj"] is not None and canonical_row["phash_obj"] is not None:
                results_df.at[idx, "duplicate_distance"] = int(row["phash_obj"] - canonical_row["phash_obj"])

            cross_split_duplicate = int(row["split"] != canonical_row["split"])
            cross_class_duplicate = int(row["class_name"] != canonical_row["class_name"])

            results_df.at[idx, "cross_split_duplicate"] = cross_split_duplicate
            results_df.at[idx, "cross_class_duplicate"] = cross_class_duplicate

            if cross_class_duplicate == 1:
                results_df.at[idx, "manual_review_required"] = 1
                results_df.at[idx, "drop_reason"] = "cross_class_duplicate_manual_review"
            elif cross_split_duplicate == 1:
                results_df.at[idx, "drop_reason"] = "cross_split_duplicate_drop_test_keep_train"
            elif same_split and same_class:
                results_df.at[idx, "drop_reason"] = "same_class_duplicate_keep_higher_quality"
            else:
                results_df.at[idx, "manual_review_required"] = 1
                results_df.at[idx, "drop_reason"] = "duplicate_manual_review"

    results_df["is_valid"] = (
        (results_df["is_corrupt"] == 0) &
        (results_df["is_blurry"] == 0) &
        (results_df["too_dark"] == 0) &
        (results_df["too_bright"] == 0) &
        (results_df["low_contrast"] == 0) &
        (results_df["is_duplicate"] == 0) &
        (results_df["manual_review_required"] == 0)
    ).astype(int)

    for idx, row in results_df.iterrows():
        if row["is_valid"] == 1:
            cleaned_path = copy_valid_image(Path(row["raw_path"]), row["split"], row["class_name"])
            results_df.at[idx, "cleaned_path"] = cleaned_path

    export_df = results_df.drop(columns=["phash_obj"])

    export_df.to_csv(REPORT_BASE / "image_validation_results.csv", index=False)

    summary_df = pd.DataFrame([{
        "total_images": len(export_df),
        "valid_images": int(export_df["is_valid"].sum()),
        "corrupt_images": int(export_df["is_corrupt"].sum()),
        "blurry_images": int(export_df["is_blurry"].sum()),
        "too_dark_images": int(export_df["too_dark"].sum()),
        "too_bright_images": int(export_df["too_bright"].sum()),
        "low_contrast_images": int(export_df["low_contrast"].sum()),
        "duplicate_images": int(export_df["is_duplicate"].sum()),
        "manual_review_images": int(export_df["manual_review_required"].sum()),
        "tenengrad_threshold": tenengrad_threshold,
        "valid_ratio": round(export_df["is_valid"].mean(), 4)
    }])
    summary_df.to_csv(REPORT_BASE / "validation_summary.csv", index=False)

    split_class_summary = (
        export_df.groupby(["split", "class_name"])
        .agg(
            total_images=("image_id", "count"),
            valid_images=("is_valid", "sum"),
            blurry_images=("is_blurry", "sum"),
            dark_images=("too_dark", "sum"),
            bright_images=("too_bright", "sum"),
            low_contrast_images=("low_contrast", "sum"),
            duplicate_images=("is_duplicate", "sum"),
            manual_review_images=("manual_review_required", "sum"),
            corrupt_images=("is_corrupt", "sum")
        )
        .reset_index()
    )
    split_class_summary.to_csv(REPORT_BASE / "validation_by_split_class.csv", index=False)

    duplicate_pairs = export_df[export_df["is_duplicate"] == 1][[
        "image_id", "canonical_image_id", "duplicate_distance", "raw_path",
        "class_name", "split", "cross_split_duplicate", "cross_class_duplicate",
        "manual_review_required", "drop_reason"
    ]]
    duplicate_pairs.to_csv(REPORT_BASE / "near_duplicate_pairs.csv", index=False)

    manual_review_df = export_df[export_df["manual_review_required"] == 1][[
        "image_id", "canonical_image_id", "raw_path", "split", "class_name",
        "duplicate_distance", "cross_split_duplicate", "cross_class_duplicate",
        "drop_reason"
    ]]
    manual_review_df.to_csv(REPORT_BASE / "manual_review_duplicates.csv", index=False)

    conn.execute(f"DELETE FROM {VALIDATION_TABLE};")
    export_df.to_sql(VALIDATION_TABLE, conn, if_exists="append", index=False)
    conn.commit()
    conn.close()

    print("Validation complete.")
    print(f"Tenengrad threshold ({TENENGRAD_PERCENTILE}th percentile of train): {tenengrad_threshold:.4f}")
    print(f"Row-level report: {REPORT_BASE / 'image_validation_results.csv'}")
    print(f"Summary report: {REPORT_BASE / 'validation_summary.csv'}")
    print(f"Split/class report: {REPORT_BASE / 'validation_by_split_class.csv'}")
    print(f"Near-duplicate report: {REPORT_BASE / 'near_duplicate_pairs.csv'}")
    print(f"Manual review report: {REPORT_BASE / 'manual_review_duplicates.csv'}")
    print(f"Cleaned dataset: {CLEANED_BASE}")
    print(f"SQL table updated: {VALIDATION_TABLE}")


if __name__ == "__main__":
    main()
