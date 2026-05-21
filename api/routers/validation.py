"""GET /validation — summary and per-image validation records from DB."""

import sqlite3
from fastapi import APIRouter, Depends, HTTPException, Query

from api.dependencies import get_db
from api.schemas import DatasetVersion, ValidationRecord, ValidationSummary

router = APIRouter(prefix="/validation", tags=["Validation"])


@router.get("/summary", response_model=ValidationSummary)
def validation_summary(db: sqlite3.Connection = Depends(get_db)):
    row = db.execute(
        """
        SELECT
            COUNT(*) AS total_images,
            SUM(is_valid) AS valid_images,
            SUM(is_corrupt) AS corrupt_images,
            SUM(is_blurry) AS blurry_images,
            SUM(too_dark) AS too_dark_images,
            SUM(too_bright) AS too_bright_images,
            SUM(low_information) AS low_information_images,
            SUM(odd_size) AS odd_size_images,
            SUM(is_exact_duplicate) AS exact_duplicate_images,
            SUM(is_near_duplicate) AS near_duplicate_images,
            ROUND(AVG(CAST(is_valid AS REAL)), 4) AS valid_ratio
        FROM image_validation_results
        """
    ).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="No validation records found.")

    return ValidationSummary(**dict(row))


@router.get("/image/{image_id}", response_model=ValidationRecord)
def get_validation_record(image_id: str, db: sqlite3.Connection = Depends(get_db)):
    row = db.execute(
        """
        SELECT
            image_id, raw_path, cleaned_path, split, class_name,
            is_corrupt, is_blurry, too_dark, too_bright, low_information,
            is_exact_duplicate, is_near_duplicate,
            duplicate_of, canonical_image_id,
            quality_score, is_valid, drop_reason, validation_timestamp
        FROM image_validation_results
        WHERE image_id = ?
        """,
        (image_id,),
    ).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail=f"image_id '{image_id}' not found.")

    return ValidationRecord(**dict(row))


@router.get("/images", response_model=list[ValidationRecord])
def list_validation_records(
    split: str | None = Query(None, description="train | test"),
    class_name: str | None = Query(None, description="bathroom | bedroom | diningroom | kitchen | livingroom"),
    is_valid: int | None = Query(None, description="1 or 0"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: sqlite3.Connection = Depends(get_db),
):
    conditions = []
    params = []

    if split is not None:
        conditions.append("split = ?")
        params.append(split)
    if class_name is not None:
        conditions.append("class_name = ?")
        params.append(class_name)
    if is_valid is not None:
        conditions.append("is_valid = ?")
        params.append(is_valid)

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    params.extend([limit, offset])

    rows = db.execute(
        f"""
        SELECT
            image_id, raw_path, cleaned_path, split, class_name,
            is_corrupt, is_blurry, too_dark, too_bright, low_information,
            is_exact_duplicate, is_near_duplicate,
            duplicate_of, canonical_image_id,
            quality_score, is_valid, drop_reason, validation_timestamp
        FROM image_validation_results
        {where}
        ORDER BY split, class_name, image_id
        LIMIT ? OFFSET ?
        """,
        params,
    ).fetchall()

    return [ValidationRecord(**dict(row)) for row in rows]


@router.get("/dataset-versions", response_model=list[DatasetVersion])
def list_dataset_versions(db: sqlite3.Connection = Depends(get_db)):
    rows = db.execute(
        """
        SELECT
            version_id AS id, version_tag, raw_image_count, cleaned_image_count,
            git_commit, created_at, notes
        FROM dataset_versions
        ORDER BY created_at DESC
        """
    ).fetchall()

    return [DatasetVersion(**dict(row)) for row in rows]
