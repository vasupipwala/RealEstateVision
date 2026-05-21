from pathlib import Path
import sqlite3
import pandas as pd

PARQUET_PATH = "./data/processed/metadata/spark/clean_metadata_parquet"
DB_PATH = Path("./db/realestatevision.db")
TABLE_NAME = "images"


def create_table(conn):
    conn.execute(f"""
    CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
        image_id TEXT PRIMARY KEY,
        source_dataset TEXT,
        split TEXT,
        class_name TEXT,
        label_id INTEGER,
        original_relative_path TEXT,
        raw_path TEXT,
        original_filename TEXT,
        new_filename TEXT,
        width INTEGER,
        height INTEGER,
        channels INTEGER,
        image_mode TEXT,
        image_format TEXT,
        aspect_ratio REAL,
        pixel_count INTEGER,
        exif_tag_count INTEGER,
        file_size_bytes INTEGER,
        file_size_kb REAL,
        file_size_mb REAL,
        md5 TEXT,
        ingestion_timestamp TEXT,
        is_small_image INTEGER,
        is_odd_aspect INTEGER,
        is_non_rgb INTEGER,
        is_missing_core_metadata INTEGER,
        invalid_split INTEGER,
        invalid_class INTEGER,
        is_unexpected_extension INTEGER,
        is_zero_or_negative_size INTEGER
    );
    """)

    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_images_split ON {TABLE_NAME}(split);")
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_images_class_name ON {TABLE_NAME}(class_name);")
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_images_label_id ON {TABLE_NAME}(label_id);")
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_images_md5 ON {TABLE_NAME}(md5);")
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_images_split_class ON {TABLE_NAME}(split, class_name);")


def main():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading Parquet from: {PARQUET_PATH}")
    df = pd.read_parquet(PARQUET_PATH)

    print(f"Loaded {len(df)} rows.")

    conn = sqlite3.connect(DB_PATH)

    try:
        create_table(conn)

        conn.execute(f"DELETE FROM {TABLE_NAME};")

        df.to_sql(TABLE_NAME, conn, if_exists="append", index=False)

        conn.commit()

        total_rows = conn.execute(f"SELECT COUNT(*) FROM {TABLE_NAME};").fetchone()[0]
        print(f"Inserted {total_rows} rows into {DB_PATH} -> table `{TABLE_NAME}`")

        sample = pd.read_sql_query(
            f"SELECT split, class_name, COUNT(*) as image_count FROM {TABLE_NAME} GROUP BY split, class_name ORDER BY split, class_name;",
            conn
        )
        print("\nCounts by split and class:")
        print(sample)

    finally:
        conn.close()


if __name__ == "__main__":
    main()
