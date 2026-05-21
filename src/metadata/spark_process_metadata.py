import os
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, LongType, DoubleType
)
from pyspark.sql.functions import col, when, lit, count, avg, round as spark_round
from pyspark.sql.functions import sum as spark_sum

# Ensure output directories exist
os.makedirs("data/processed/metadata/spark", exist_ok=True)
os.makedirs("data/analytics/quality", exist_ok=True)

def main():
    # 1. Initialize local Spark session
    spark = (
        SparkSession.builder
        .appName("RealEstateVision-Metadata")
        .master("local[*]")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )

    # 2. Define explicit schema to match our Python extraction script
    schema = StructType([
        StructField("image_id", StringType(), True),
        StructField("source_dataset", StringType(), True),
        StructField("split", StringType(), True),
        StructField("class_name", StringType(), True),
        StructField("label_id", IntegerType(), True),
        StructField("original_relative_path", StringType(), True),
        StructField("raw_path", StringType(), True),
        StructField("original_filename", StringType(), True),
        StructField("new_filename", StringType(), True),
        StructField("width", IntegerType(), True),
        StructField("height", IntegerType(), True),
        StructField("channels", IntegerType(), True),
        StructField("image_mode", StringType(), True),
        StructField("image_format", StringType(), True),
        StructField("aspect_ratio", DoubleType(), True),
        StructField("pixel_count", LongType(), True),
        StructField("exif_tag_count", IntegerType(), True),
        StructField("file_size_bytes", LongType(), True),
        StructField("md5", StringType(), True),
        StructField("ingestion_timestamp", StringType(), True),
    ])

    # 3. Read the extracted CSV
    print("Reading extracted metadata...")
    input_path = "./data/processed/metadata/extracted/mit_indoor_subset_index.csv"
    df = spark.read.option("header", True).schema(schema).csv(input_path)

    # 4. Transformations (Feature Engineering) : Metadata Anomaly Detection & Cleaning
    # Flagging things we can catch without looking at pixels
    print("Applying transformations...")
    clean_df = (
        df
        .withColumn("file_size_kb", spark_round(col("file_size_bytes") / 1024, 2))
        .withColumn("is_small_image", when((col("width") < 256) | (col("height") < 256), lit(1)).otherwise(lit(0)))
        .withColumn("is_odd_aspect", when((col("aspect_ratio") > 2.5) | (col("aspect_ratio") < 0.4), lit(1)).otherwise(lit(0)))
        .withColumn("is_non_rgb", when(col("image_mode") != "RGB", lit(1)).otherwise(lit(0)))
    )

    # 5. Analytics: Class Distribution & Average Sizes
    print("Computing quality analytics...")
    class_summary = (
        clean_df.groupBy("split", "class_name")
        .agg(
            count("*").alias("image_count"),
            spark_round(avg("width"), 1).alias("avg_width"),
            spark_round(avg("height"), 1).alias("avg_height"),
            spark_round(avg("file_size_kb"), 1).alias("avg_file_size_kb")
        )
        .orderBy("split", "class_name")
    )

    # 6. Analytics: High-level Metadata Anomaly Summary
    anomaly_summary = (
        clean_df.groupBy("split")
        .agg(
            count("*").alias("total_images"),
            count(when(col("is_small_image") == 1, True)).alias("small_images"),
            count(when(col("is_odd_aspect") == 1, True)).alias("odd_aspect_images"),
            count(when(col("is_non_rgb") == 1, True)).alias("non_rgb_images")
        )
    )

    # 7. Analytics: Exact File Duplicates (MD5 collisions)
    duplicate_summary = (
        clean_df.groupBy("md5")
        .agg(count("*").alias("count"),)
        .filter(col("count") > 1)
    )
    
    # 8. Null summary: Missing-value counts per column
    null_summary = clean_df.select([
        spark_sum(col(c).isNull().cast("int")).alias(c)
        for c in ["width", "height", "channels", "image_mode", "image_format", "aspect_ratio", "md5"]
        ])
    
    #  9. Per-class anomaly summary : Class-level anomaly rates and counts
    class_anomaly_summary = (
        clean_df.groupBy("class_name")
        .agg(
            count("*").alias("total_images"),
            count(when(col("is_small_image") == 1, True)).alias("small_images"),
            count(when(col("is_odd_aspect") == 1, True)).alias("odd_aspect_images"),
            count(when(col("is_non_rgb") == 1, True)).alias("non_rgb_images")
        )
        .orderBy("class_name")
    )

    # 10. Categorical consistency snapshots : Distinct value checks for split, class, mode, format
    split_summary = clean_df.groupBy("split").count().orderBy("split")
    format_summary = clean_df.groupBy("image_format").count().orderBy(col("count").desc())
    mode_summary = clean_df.groupBy("image_mode").count().orderBy(col("count").desc())
    
    
    # 11. Save Outputs
    print("Writing outputs to disk...")
    
    # A. Save the enriched master table as Parquet for fast querying later
    parquet_out = "./data/processed/metadata/spark/clean_metadata_parquet"
    clean_df.write.mode("overwrite").parquet(parquet_out)

    # B. Save analytics summaries as CSVs so you can read them easily
    class_summary.coalesce(1).write.mode("overwrite").option("header", True).csv("./data/analytics/quality/class_summary")
    anomaly_summary.coalesce(1).write.mode("overwrite").option("header", True).csv("./data/analytics/quality/anomaly_summary")
    duplicate_summary.coalesce(1).write.mode("overwrite").option("header", True).csv("./data/analytics/quality/duplicate_summary")
    null_summary.coalesce(1).write.mode("overwrite").option("header", True).csv("./data/analytics/quality/null_summary")
    class_anomaly_summary.coalesce(1).write.mode("overwrite").option("header", True).csv("./data/analytics/quality/class_anomaly_summary")
    split_summary.coalesce(1).write.mode("overwrite").option("header", True).csv("./data/analytics/quality/split_summary")
    format_summary.coalesce(1).write.mode("overwrite").option("header", True).csv("./data/analytics/quality/format_summary")
    mode_summary.coalesce(1).write.mode("overwrite").option("header", True).csv("./data/analytics/quality/mode_summary")


    print(f"PySpark processing complete.")
    print(f"- Cleaned Parquet saved to: {parquet_out}")
    print(f"- Analytics CSVs saved to: ./data/analytics/quality/")
    
    spark.stop()

if __name__ == "__main__":
    main()
