"""
Data Versioning — Record Dataset Version in SQL
Run this after dvc add + git commit to register the dataset state in the DB.

Usage:
    python output/record_dataset_version.py \
        --db-path db/realestatevision.db \
        --version-tag cleaned_v1 \
        --notes "First validated and DVC-tracked dataset"
"""

import argparse
import hashlib
import json
import re
import sqlite3
import subprocess
from datetime import datetime
from pathlib import Path

# ── Paths ────────────────────────────────────────────────
DB_PATH = "./db/realestatevision.db"
RAW_PATH = Path("./data/raw")
CLEANED_PATH = Path("./data/processed/cleaned")
REPORTS_PATH = Path("./data/processed/validation_reports")
DVC_FILES = {
    "raw": Path("./data/raw.dvc"),
    "cleaned": Path("./data/processed/cleaned.dvc"),
    "validation_reports": Path("./data/processed/validation_reports.dvc"),
}


# ── Helpers ──────────────────────────────────────────────
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_manifest_hash(directories: list[Path]) -> str:
    """Compute a single SHA256 hash over all files in the given directories."""
    entries = []
    for d in directories:
        if d.exists():
            for p in sorted(d.rglob("*")):
                if p.is_file():
                    entries.append({
                        "path": str(p),
                        "size": p.stat().st_size,
                        "sha256": sha256_file(p),
                    })
    manifest_json = json.dumps(entries, sort_keys=True)
    return hashlib.sha256(manifest_json.encode()).hexdigest()


def read_dvc_md5(dvc_file: Path) -> str | None:
    if not dvc_file.exists():
        return None
    content = dvc_file.read_text()
    m = re.search(r"md5:\s*([a-fA-F0-9]+)", content)
    return m.group(1) if m else None


def git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def count_images(directory: Path) -> int:
    if not directory.exists():
        return 0
    return sum(1 for p in directory.rglob("*")
               if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"})


# ── SQL ──────────────────────────────────────────────────
def ensure_table(conn: sqlite3.Connection):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS dataset_versions (
            version_id          INTEGER PRIMARY KEY AUTOINCREMENT,
            version_tag         TEXT    NOT NULL UNIQUE,
            raw_image_count     INTEGER,
            cleaned_image_count INTEGER,
            raw_dvc_md5         TEXT,
            cleaned_dvc_md5     TEXT,
            reports_dvc_md5     TEXT,
            manifest_sha256     TEXT,
            git_commit          TEXT,
            created_at          TEXT    NOT NULL,
            notes               TEXT
        );
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_dv_version_tag
        ON dataset_versions(version_tag);
    """)
    conn.commit()


def insert_version(conn: sqlite3.Connection, record: dict):
    conn.execute("""
        INSERT OR REPLACE INTO dataset_versions (
            version_tag, raw_image_count, cleaned_image_count,
            raw_dvc_md5, cleaned_dvc_md5, reports_dvc_md5,
            manifest_sha256, git_commit, created_at, notes
        ) VALUES (
            :version_tag, :raw_image_count, :cleaned_image_count,
            :raw_dvc_md5, :cleaned_dvc_md5, :reports_dvc_md5,
            :manifest_sha256, :git_commit, :created_at, :notes
        )
    """, record)
    conn.commit()


# ── Main ─────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Record a dataset version in SQLite.")
    parser.add_argument("--db-path", default=DB_PATH)
    parser.add_argument("--version-tag", required=True,
                        help="e.g. cleaned_v1, cleaned_v2")
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    print(f"Recording dataset version: {args.version_tag}")

    conn = sqlite3.connect(args.db_path)
    ensure_table(conn)

    record = {
        "version_tag": args.version_tag,
        "raw_image_count": count_images(RAW_PATH),
        "cleaned_image_count": count_images(CLEANED_PATH),
        "raw_dvc_md5": read_dvc_md5(DVC_FILES["raw"]),
        "cleaned_dvc_md5": read_dvc_md5(DVC_FILES["cleaned"]),
        "reports_dvc_md5": read_dvc_md5(DVC_FILES["validation_reports"]),
        "manifest_sha256": compute_manifest_hash([CLEANED_PATH, REPORTS_PATH]),
        "git_commit": git_head(),
        "created_at": datetime.utcnow().isoformat(),
        "notes": args.notes,
    }

    insert_version(conn, record)
    conn.close()

    print("\n✅ Dataset version recorded:")
    print(json.dumps(record, indent=2))

    print("\nQuery dataset_versions table:")
    print("  sqlite3 db/realestatevision.db \"SELECT * FROM dataset_versions;\"")


if __name__ == "__main__":
    main()
