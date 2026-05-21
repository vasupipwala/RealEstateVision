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

IMAGE_ROOT = Path("./data/raw/mit_indoor_subset")

# Conservative thresholds: only extreme cases
ISSUE_TYPES = {
    "dark": {"threshold": 0.05},
    "light": {"threshold": 0.98},
    "blurry": {"threshold": 0.99},
    "low_information": {"threshold": 0.99},
    "odd_aspect_ratio": {"threshold": 0.01},
    "odd_size": {"threshold": 0.01},
}
# duplicates / near_duplicates are handled from CleanVision issue outputs directly


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


def copy_valid_image(src_path: Path, split: str, class_name: str):
    target_dir = CLEANED_BASE / split / class_name
    target_dir.mkdir(parents=True, exist_ok=True)
    dst_path = target_dir / src_path.name
    shutil.copy2(src_path, dst_path)
    return str(dst_path)


def split_priority(split_name):
    return 1 if split_name == "train" else 0


def choose_canonical(group_df):
    candidate_df = group_df.copy()
    candidate_df = candidate_df.sort_values(
        by=["split_priority", "quality_score", "image_id"],
        ascending=[False, False, True]
    )
    return candidate_df.iloc[0]["image_id"]


def relative_path_from_root(raw_path: str):
    return str(Path(raw_path).relative_to(IMAGE_ROOT))


def build_issue_lookup(issue_df, path_col="file"):
    if issue_df is None or len(issue_df) == 0:
        return set()
    return set(issue_df[path_col].astype(str).tolist())


def normalize_duplicate_pairs(df, left_col, right_col, score_col=None, pair_type="near"):
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["file_a", "file_b", "distance", "pair_type"])

    out = df.copy()
    out = out.rename(columns={left_col: "file_a", right_col: "file_b"})
    if score_col and score_col in out.columns:
        out["distance"] = out[score_col]
    else:
        out["distance"] = None
    out["pair_type"] = pair_type
    return out[["file_a", "file_b", "distance", "pair_type"]]


def main():
    conn = sqlite3.connect(DB_PATH)
    ensure_validation_table(conn)

    images_df = fetch_images_metadata(conn)

    # Build base records
    rows = []
    for _, row in images_df.iterrows():
        rows.append({
            "image_id": row["image_id"],
            "raw_path": row["raw_path"],
            "relative_file": relative_path_from_root(row["raw_path"]),
            "split": row["split"],
            "class_name": row["class_name"],
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

    # Run CleanVision
    imagelab = Imagelab(data_path=str(IMAGE_ROOT))
    imagelab.find_issues(issue_types=ISSUE_TYPES)

    issues = imagelab.issues if hasattr(imagelab, "issues") else {}

    dark_files = build_issue_lookup(issues.get("dark"))
    light_files = build_issue_lookup(issues.get("light"))
    blurry_files = build_issue_lookup(issues.get("blurry"))
    low_info_files = build_issue_lookup(issues.get("low_information"))
    odd_aspect_files = build_issue_lookup(issues.get("odd_aspect_ratio"))
    odd_size_files = build_issue_lookup(issues.get("odd_size"))

    # Optional image-level scores from imagelab if available
    issue_summary = imagelab.issue_summary
    issue_scores = imagelab.info if hasattr(imagelab, "info") else None

    for idx, row in results_df.iterrows():
        rf = row["relative_file"]
        results_df.at[idx, "too_dark"] = int(rf in dark_files)
        results_df.at[idx, "too_bright"] = int(rf in light_files)
        results_df.at[idx, "is_blurry"] = int(rf in blurry_files)
        results_df.at[idx, "low_information"] = int(rf in low_info_files)
        results_df.at[idx, "odd_aspect_ratio"] = int(rf in odd_aspect_files)
        results_df.at[idx, "odd_size"] = int(rf in odd_size_files)

        # Conservative quality score for retention policy
        penalty = (
            3 * results_df.at[idx, "is_corrupt"] +
            2 * results_df.at[idx, "is_blurry"] +
            1 * results_df.at[idx, "too_dark"] +
            1 * results_df.at[idx, "too_bright"] +
            1 * results_df.at[idx, "low_information"] +
            1 * results_df.at[idx, "odd_aspect_ratio"] +
            1 * results_df.at[idx, "odd_size"]
        )
        results_df.at[idx, "quality_score"] = float(-penalty)

    # Get duplicates / near duplicates from CleanVision
    exact_dup_df = issues.get("exact_duplicates")
    near_dup_df = issues.get("near_duplicates")

    exact_pairs = normalize_duplicate_pairs(exact_dup_df, "file", "duplicate", None, "exact")
    near_pairs = normalize_duplicate_pairs(near_dup_df, "file", "duplicate", "distance", "near")

    duplicate_pairs = pd.concat([exact_pairs, near_pairs], ignore_index=True) if (
        len(exact_pairs) or len(near_pairs)
    ) else pd.DataFrame(columns=["file_a", "file_b", "distance", "pair_type"])

    # Union-Find over duplicate graph
    file_to_idx = {f: i for i, f in enumerate(results_df["relative_file"].tolist())}
    parent = list(range(len(results_df)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for _, pair in duplicate_pairs.iterrows():
        a = pair["file_a"]
        b = pair["file_b"]
        if a in file_to_idx and b in file_to_idx:
            union(file_to_idx[a], file_to_idx[b])

    groups = {}
    for i in range(len(results_df)):
        root = find(i)
        groups.setdefault(root, []).append(i)

    duplicate_groups = [g for g in groups.values() if len(g) > 1]

    pair_lookup = {}
    for _, pair in duplicate_pairs.iterrows():
        key = tuple(sorted([pair["file_a"], pair["file_b"]]))
        pair_lookup[key] = {
            "distance": pair["distance"],
            "pair_type": pair["pair_type"]
        }

    for group_indices in duplicate_groups:
        group_df = results_df.iloc[group_indices].copy()
        group_df["split_priority"] = group_df["split"].apply(split_priority)

        canonical_image_id = choose_canonical(group_df)
        canonical_row = group_df[group_df["image_id"] == canonical_image_id].iloc[0]

        has_train = (group_df["split"] == "train").any()
        same_split = len(group_df["split"].unique()) == 1
        same_class = len(group_df["class_name"].unique()) == 1
        cross_class = len(group_df["class_name"].unique()) > 1

        for idx in group_indices:
            row = results_df.loc[idx]
            results_df.at[idx, "canonical_image_id"] = canonical_image_id

            if row["image_id"] == canonical_image_id:
                continue

            key = tuple(sorted([row["relative_file"], canonical_row["relative_file"]]))
            pair_meta = pair_lookup.get(key, {})
            pair_type = pair_meta.get("pair_type")
            distance = pair_meta.get("distance")

            if pair_type == "exact":
                results_df.at[idx, "is_exact_duplicate"] = 1
            else:
                results_df.at[idx, "is_near_duplicate"] = 1

            results_df.at[idx, "duplicate_of"] = canonical_image_id
            results_df.at[idx, "duplicate_distance"] = distance
            results_df.at[idx, "cross_split_duplicate"] = int(row["split"] != canonical_row["split"])
            results_df.at[idx, "cross_class_duplicate"] = int(row["class_name"] != canonical_row["class_name"])

            # Policy
            if has_train and row["split"] == "test":
                results_df.at[idx, "drop_reason"] = "drop_test_duplicate_train_priority"

            elif same_split and same_class:
                results_df.at[idx, "drop_reason"] = "same_split_same_class_keep_higher_quality"

            elif cross_class and same_split:
                results_df.at[idx, "manual_review_required"] = 1
                results_df.at[idx, "drop_reason"] = "kept_cross_class_duplicate_same_split"

            else:
                results_df.at[idx, "drop_reason"] = "drop_noncanonical_duplicate"

    # Final validity
    duplicate_drop_mask = results_df["drop_reason"].isin([
        "drop_test_duplicate_train_priority",
        "same_split_same_class_keep_higher_quality",
        "drop_noncanonical_duplicate"
    ])

    results_df["is_valid"] = (
        (results_df["is_corrupt"] == 0) &
        (results_df["is_blurry"] == 0) &
        (results_df["too_dark"] == 0) &
        (results_df["too_bright"] == 0) &
        (results_df["low_information"] == 0) &
        (~duplicate_drop_mask)
    ).astype(int)

    for idx, row in results_df.iterrows():
        if row["is_valid"] == 1:
            cleaned_path = copy_valid_image(Path(row["raw_path"]), row["split"], row["class_name"])
            results_df.at[idx, "cleaned_path"] = cleaned_path

    export_df = results_df.drop(columns=["relative_file"])

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
        (export_df["is_exact_duplicate"] == 1) |
        (export_df["is_near_duplicate"] == 1) |
        (export_df["drop_reason"] == "kept_cross_class_duplicate_same_split")
    ][[
        "image_id", "duplicate_of", "canonical_image_id", "duplicate_distance",
        "raw_path", "class_name", "split", "cross_split_duplicate",
        "cross_class_duplicate", "manual_review_required", "drop_reason"
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
