from pathlib import Path
from datetime import datetime
import shutil
import sqlite3
import pandas as pd
from PIL import Image, UnidentifiedImageError
from cleanvision import Imagelab

DB_PATH = "./db/realestatevision.db"
IMAGES_TABLE = "images"
VALIDATION_TABLE = "image_validation_results"

CLEANED_BASE = Path("./data/processed/cleaned")
REPORT_BASE = Path("./data/processed/validation_reports")
REPORT_BASE.mkdir(parents=True, exist_ok=True)
CLEANED_BASE.mkdir(parents=True, exist_ok=True)

IMAGE_ROOT = Path("./data/raw/mit_indoor_subset").resolve()


def ensure_validation_table(conn):
    conn.execute(f"""
    CREATE TABLE IF NOT EXISTS {VALIDATION_TABLE} (
        image_id TEXT PRIMARY KEY,
        raw_path TEXT,
        cleaned_path TEXT,
        split TEXT,
        class_name TEXT,
        is_corrupt INTEGER,
        is_blurry INTEGER,
        too_dark INTEGER,
        too_bright INTEGER,
        low_information INTEGER,
        odd_aspect_ratio INTEGER,
        odd_size INTEGER,
        is_exact_duplicate INTEGER,
        is_near_duplicate INTEGER,
        canonical_image_id TEXT,
        duplicate_of TEXT,
        duplicate_distance REAL,
        cross_split_duplicate INTEGER,
        cross_class_duplicate INTEGER,
        manual_review_required INTEGER,
        drop_reason TEXT,
        quality_score REAL,
        blurry_score REAL,
        dark_score REAL,
        light_score REAL,
        low_information_score REAL,
        odd_aspect_ratio_score REAL,
        odd_size_score REAL,
        exact_duplicates_score REAL,
        near_duplicates_score REAL,
        is_valid INTEGER,
        validation_timestamp TEXT,
        FOREIGN KEY(image_id) REFERENCES images(image_id)
    );
    """)

    existing_cols = {row[1] for row in conn.execute(f"PRAGMA table_info({VALIDATION_TABLE});").fetchall()}
    extra_columns = {
        "cleaned_path": "TEXT",
        "low_information": "INTEGER",
        "odd_aspect_ratio": "INTEGER",
        "odd_size": "INTEGER",
        "is_exact_duplicate": "INTEGER",
        "is_near_duplicate": "INTEGER",
        "canonical_image_id": "TEXT",
        "duplicate_of": "TEXT",
        "duplicate_distance": "REAL",
        "cross_split_duplicate": "INTEGER",
        "cross_class_duplicate": "INTEGER",
        "manual_review_required": "INTEGER",
        "drop_reason": "TEXT",
        "quality_score": "REAL",
        "blurry_score": "REAL",
        "dark_score": "REAL",
        "light_score": "REAL",
        "low_information_score": "REAL",
        "odd_aspect_ratio_score": "REAL",
        "odd_size_score": "REAL",
        "exact_duplicates_score": "REAL",
        "near_duplicates_score": "REAL",
    }

    for col_name, col_type in extra_columns.items():
        if col_name not in existing_cols:
            conn.execute(f"ALTER TABLE {VALIDATION_TABLE} ADD COLUMN {col_name} {col_type};")

    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_validation_split ON {VALIDATION_TABLE}(split);")
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_validation_class_name ON {VALIDATION_TABLE}(class_name);")
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_validation_validity ON {VALIDATION_TABLE}(is_valid);")
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_validation_canonical_image_id ON {VALIDATION_TABLE}(canonical_image_id);")


def fetch_images_metadata(conn):
    query = f"""
    SELECT image_id, raw_path, split, class_name
    FROM {IMAGES_TABLE}
    ORDER BY split, class_name, image_id;
    """
    return pd.read_sql_query(query, conn)


def verify_corrupt(image_path: str):
    try:
        with Image.open(image_path) as img:
            img.verify()
        return 0
    except (UnidentifiedImageError, OSError, ValueError):
        return 1


def get_image_size(image_path: str):
    try:
        with Image.open(image_path) as img:
            return img.width, img.height
    except Exception:
        return None, None


def copy_valid_image(src_path: Path, split: str, class_name: str):
    target_dir = CLEANED_BASE / split / class_name
    target_dir.mkdir(parents=True, exist_ok=True)
    dst_path = target_dir / src_path.name
    shutil.copy2(src_path, dst_path)
    return str(dst_path)


def split_priority(split_name):
    return 1 if split_name == "train" else 0


def relative_path_from_root(raw_path: str):
    return str(Path(raw_path).resolve().relative_to(IMAGE_ROOT)).replace("\\", "/")


def normalize_duplicate_member(x):
    p = Path(str(x)).resolve()
    try:
        return str(p.relative_to(IMAGE_ROOT)).replace("\\", "/")
    except ValueError:
        return str(p).replace("\\", "/")


def normalize_dup_set(dup_set):
    normalized = []
    for x in dup_set:
        if isinstance(x, int):
            normalized.append(x)
        else:
            normalized.append(normalize_duplicate_member(x))
    return normalized


def build_quality_score(row):
    penalty = 0.0

    penalty += 10.0 * float(row.get("is_corrupt", 0) or 0)
    penalty += 5.0 * float(row.get("is_blurry", 0) or 0)
    penalty += 2.5 * float(row.get("too_dark", 0) or 0)
    penalty += 2.5 * float(row.get("too_bright", 0) or 0)
    penalty += 5.0 * float(row.get("low_information", 0) or 0)
    #penalty += 1.0 * float(row.get("odd_aspect_ratio", 0) or 0)
    #penalty += 1.0 * float(row.get("odd_size", 0) or 0)

    width = row.get("width", None)
    height = row.get("height", None)
    resolution_bonus = 0.0
#    if pd.notna(width) and pd.notna(height):
#        resolution_bonus = min((float(width) * float(height)) / 1_000_000.0, 10.0) * 0.05

    return -penalty + resolution_bonus


def choose_canonical(group_df):
    candidate_df = group_df.copy()
    candidate_df = candidate_df.sort_values(
        by=["split_priority", "quality_score", "width", "height", "image_id"],
        ascending=[False, False, False, False, True]
    )
    return candidate_df.iloc[0]["image_id"]


def check_validity(row):
    if int(row.get("is_corrupt", 0) or 0) == 1:
        return 0

    drop_reason = row.get("drop_reason", None)
    if pd.notna(drop_reason) and str(drop_reason).startswith("drop_"):
        return 0

    if row["split"] == "train":
        if (
            int(row.get("is_blurry", 0) or 0) == 1 or
            int(row.get("too_dark", 0) or 0) == 1 or
            int(row.get("too_bright", 0) or 0) == 1 or
            int(row.get("low_information", 0) or 0) == 1
        ):
            return 0

    return 1
    

def main():
    conn = sqlite3.connect(DB_PATH)
    ensure_validation_table(conn)

    images_df = fetch_images_metadata(conn)

    rows = []
    for _, row in images_df.iterrows():
        width, height = get_image_size(row["raw_path"])
        rows.append({
            "image_id": row["image_id"],
            "raw_path": row["raw_path"],
            "relative_file": relative_path_from_root(row["raw_path"]),
            "split": row["split"],
            "class_name": row["class_name"],
            "width": width,
            "height": height,
            "is_corrupt": verify_corrupt(row["raw_path"]),
            "is_blurry": 0,
            "too_dark": 0,
            "too_bright": 0,
            "low_information": 0,
            "odd_aspect_ratio": 0,
            "odd_size": 0,
            "is_exact_duplicate": 0,
            "is_near_duplicate": 0,
            "canonical_image_id": None,
            "duplicate_of": None,
            "duplicate_distance": None,
            "cross_split_duplicate": 0,
            "cross_class_duplicate": 0,
            "manual_review_required": 0,
            "drop_reason": None,
            "quality_score": 0.0,
            "cleaned_path": None,
            "is_valid": 0,
            "validation_timestamp": datetime.utcnow().isoformat(),
        })

    results_df = pd.DataFrame(rows)

    imagelab = Imagelab(data_path=str(IMAGE_ROOT))
    imagelab.find_issues()

    issues_df = imagelab.issues.copy() if hasattr(imagelab, "issues") else pd.DataFrame()
    info = imagelab.info if hasattr(imagelab, "info") else {}
    issue_summary = imagelab.issue_summary if hasattr(imagelab, "issue_summary") else pd.DataFrame()

    print("INFO KEYS:", info.keys() if isinstance(info, dict) else type(info))
    print("EXACT DUP INFO:", info.get("exact_duplicates", {}) if isinstance(info, dict) else "N/A")
    print("NEAR DUP INFO:", info.get("near_duplicates", {}) if isinstance(info, dict) else "N/A")
    print("Detected CleanVision issue columns:", list(issues_df.columns))
    if not issue_summary.empty:
        print(issue_summary)

    if not issues_df.empty:
        issues_df = issues_df.reset_index(drop=True)
        image_file_order = results_df[["relative_file"]].copy().reset_index(drop=True)

        if len(issues_df) != len(image_file_order):
            raise ValueError(
                f"Mismatch between CleanVision issues rows ({len(issues_df)}) "
                f"and metadata rows ({len(image_file_order)})."
            )

        issues_df["relative_file"] = image_file_order["relative_file"].astype(str).str.replace("\\", "/", regex=False)

        merge_cols = [
            "relative_file",
            "is_blurry_issue",
            "is_dark_issue",
            "is_light_issue",
            "is_low_information_issue",
            "is_odd_aspect_ratio_issue",
            "is_odd_size_issue",
            "is_exact_duplicates_issue",
            "is_near_duplicates_issue",
            "blurry_score",
            "dark_score",
            "light_score",
            "low_information_score",
            "odd_aspect_ratio_score",
            "odd_size_score",
            "exact_duplicates_score",
            "near_duplicates_score",
        ]
        available_cols = [c for c in merge_cols if c in issues_df.columns]
        issues_df = issues_df[available_cols]

        results_df = results_df.merge(issues_df, on="relative_file", how="left", validate="1:1")

        flag_map = [
            ("is_blurry_issue", "is_blurry"),
            ("is_dark_issue", "too_dark"),
            ("is_light_issue", "too_bright"),
            ("is_low_information_issue", "low_information"),
            ("is_odd_aspect_ratio_issue", "odd_aspect_ratio"),
            ("is_odd_size_issue", "odd_size"),
            ("is_exact_duplicates_issue", "is_exact_duplicate"),
            ("is_near_duplicates_issue", "is_near_duplicate"),
        ]

        for src_col, dst_col in flag_map:
            if src_col in results_df.columns:
                results_df[dst_col] = results_df[src_col].fillna(False).eq(True).astype(int)

        for score_col in [
            "blurry_score",
            "dark_score",
            "light_score",
            "low_information_score",
            "odd_aspect_ratio_score",
            "odd_size_score",
            "exact_duplicates_score",
            "near_duplicates_score",
        ]:
            if score_col not in results_df.columns:
                results_df[score_col] = None
    else:
        print("Warning: imagelab.issues is empty.")
        for score_col in [
            "blurry_score",
            "dark_score",
            "light_score",
            "low_information_score",
            "odd_aspect_ratio_score",
            "odd_size_score",
            "exact_duplicates_score",
            "near_duplicates_score",
        ]:
            results_df[score_col] = None

    results_df["quality_score"] = results_df.apply(build_quality_score, axis=1)
    results_df["split_priority"] = results_df["split"].apply(split_priority)

    exact_sets = info.get("exact_duplicates", {}).get("sets", []) if isinstance(info, dict) else []
    near_sets = info.get("near_duplicates", {}).get("sets", []) if isinstance(info, dict) else []

    duplicate_groups_files = []

    for dup_set in exact_sets:
        normalized = normalize_dup_set(dup_set)
        if len(normalized) > 1:
            duplicate_groups_files.append(("exact", normalized))

    for dup_set in near_sets:
        normalized = normalize_dup_set(dup_set)
        if len(normalized) > 1:
            duplicate_groups_files.append(("near", normalized))

    print("Exact duplicate sets found:", len(exact_sets))
    print("Near duplicate sets found:", len(near_sets))
    print("Normalized duplicate groups:", duplicate_groups_files)

    file_to_idx = {f: i for i, f in enumerate(results_df["relative_file"].tolist())}

    for pair_type, dup_files in duplicate_groups_files:
        valid_indices = []

        for f in dup_files:
            if isinstance(f, int):
                if 0 <= f < len(results_df):
                    valid_indices.append(f)
            elif f in file_to_idx:
                valid_indices.append(file_to_idx[f])

        valid_indices = sorted(set(valid_indices))

        if len(valid_indices) <= 1:
            continue

        group_df = results_df.loc[valid_indices].copy()
        canonical_image_id = choose_canonical(group_df)
        canonical_row = group_df[group_df["image_id"] == canonical_image_id].iloc[0]

        has_train = (group_df["split"] == "train").any()
        same_split = len(group_df["split"].unique()) == 1
        same_class = len(group_df["class_name"].unique()) == 1
        cross_class = len(group_df["class_name"].unique()) > 1

        for idx in valid_indices:
            row = results_df.loc[idx]
            results_df.at[idx, "canonical_image_id"] = canonical_image_id

            if row["image_id"] == canonical_image_id:
                continue

            results_df.at[idx, "duplicate_of"] = canonical_image_id
            results_df.at[idx, "duplicate_distance"] = (
                float(row.get("near_duplicates_score")) if pair_type == "near" and pd.notna(row.get("near_duplicates_score")) else 0.0
            )
            results_df.at[idx, "cross_split_duplicate"] = int(row["split"] != canonical_row["split"])
            results_df.at[idx, "cross_class_duplicate"] = int(row["class_name"] != canonical_row["class_name"])

            if pair_type == "exact":
                results_df.at[idx, "is_exact_duplicate"] = 1
            else:
                results_df.at[idx, "is_near_duplicate"] = 1

            if has_train and row["split"] == "test":
                results_df.at[idx, "drop_reason"] = "drop_test_duplicate_train_priority"
            elif same_split and same_class:
                results_df.at[idx, "drop_reason"] = "drop_same_split_same_class_lower_quality"
            elif cross_class and same_split:
                results_df.at[idx, "manual_review_required"] = 1
                results_df.at[idx, "drop_reason"] = "keep_cross_class_duplicate_same_split_manual_review"
            else:
                results_df.at[idx, "drop_reason"] = "drop_noncanonical_duplicate"

    results_df["quality_score"] = results_df.apply(build_quality_score, axis=1)
    results_df["is_valid"] = results_df.apply(check_validity, axis=1)

    for idx, row in results_df.iterrows():
        if row["is_valid"] == 1:
            cleaned_path = copy_valid_image(Path(row["raw_path"]), row["split"], row["class_name"])
            results_df.at[idx, "cleaned_path"] = cleaned_path

    drop_helper_cols = [
        "is_blurry_issue",
        "is_dark_issue",
        "is_light_issue",
        "is_low_information_issue",
        "is_odd_aspect_ratio_issue",
        "is_odd_size_issue",
        "is_exact_duplicates_issue",
        "is_near_duplicates_issue",
        "relative_file",
        "split_priority",
        "width",
        "height",
    ]
    export_df = results_df.drop(columns=[c for c in drop_helper_cols if c in results_df.columns], errors="ignore")

    export_df.to_csv(REPORT_BASE / "image_validation_results.csv", index=False)

    summary_df = pd.DataFrame([{
        "total_images": len(export_df),
        "valid_images": int(export_df["is_valid"].sum()),
        "corrupt_images": int(export_df["is_corrupt"].sum()),
        "blurry_images": int(export_df["is_blurry"].sum()),
        "too_dark_images": int(export_df["too_dark"].sum()),
        "too_bright_images": int(export_df["too_bright"].sum()),
        "low_information_images": int(export_df["low_information"].sum()),
        "odd_aspect_ratio_images": int(export_df["odd_aspect_ratio"].sum()),
        "odd_size_images": int(export_df["odd_size"].sum()),
        "exact_duplicate_images": int(export_df["is_exact_duplicate"].sum()),
        "near_duplicate_images": int(export_df["is_near_duplicate"].sum()),
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
            low_information_images=("low_information", "sum"),
            odd_aspect_ratio_images=("odd_aspect_ratio", "sum"),
            odd_size_images=("odd_size", "sum"),
            exact_duplicate_images=("is_exact_duplicate", "sum"),
            near_duplicate_images=("is_near_duplicate", "sum"),
            corrupt_images=("is_corrupt", "sum")
        )
        .reset_index()
    )
    split_class_summary.to_csv(REPORT_BASE / "validation_by_split_class.csv", index=False)

    duplicate_report = export_df[
        (
        (export_df["is_exact_duplicate"] == 1) |
        (export_df["is_near_duplicate"] == 1) |
        (export_df["manual_review_required"] == 1) |
        (export_df["duplicate_of"].notna())
        )
        #&
        #(export_df["duplicate_of"].apply(lambda x: isinstance(x, str) and len(x.strip()) > 0))
        #(export_df["duplicate_of"].notna())
    ][[
        "image_id",
        "duplicate_of",
        "canonical_image_id",
        "duplicate_distance",
        "raw_path",
        "class_name",
        "split",
        "cross_split_duplicate",
        "cross_class_duplicate",
        "manual_review_required",
        "drop_reason"
    ]]
    duplicate_report.to_csv(REPORT_BASE / "near_duplicate_pairs.csv", index=False)

    conn.execute(f"DELETE FROM {VALIDATION_TABLE};")
    export_df.to_sql(VALIDATION_TABLE, conn, if_exists="append", index=False)
    conn.commit()
    conn.close()

    print("Validation complete using CleanVision.")
    print(f"Row-level report: {REPORT_BASE / 'image_validation_results.csv'}")
    print(f"Summary report: {REPORT_BASE / 'validation_summary.csv'}")
    print(f"Split/class report: {REPORT_BASE / 'validation_by_split_class.csv'}")
    print(f"Duplicate report: {REPORT_BASE / 'near_duplicate_pairs.csv'}")
    print(f"Cleaned dataset: {CLEANED_BASE}")
    print(f"SQL table updated: {VALIDATION_TABLE}")


if __name__ == "__main__":
    main()
