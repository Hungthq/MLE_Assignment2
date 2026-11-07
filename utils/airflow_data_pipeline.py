"""
Bronze -> silver -> gold for ONE snapshot month (the per-month version of utils/data_pipeline_all_months.py, for Airflow).

Usage (from the project root):
    python -m utils.airflow_data_pipeline --snapshotdate 2024-09-01 --stage labels     # loans -> gold label store
    python -m utils.airflow_data_pipeline --snapshotdate 2024-09-01 --stage features   # attributes/financials/clickstream -> gold feature store

Exit code 99 = the source has no rows for that month (Airflow marks the task as skipped).
"""

import argparse
import os
import sys
from datetime import datetime

import pyspark
from pyspark.sql.functions import col

import utils.data_processing_bronze
import utils.data_processing_silver
import utils.data_processing_gold

SOURCE = {
    "lms": "data/lms_loan_daily.csv",
    "clickstream": "data/feature_clickstream.csv",
    "attributes": "data/features_attributes.csv",
    "financials": "data/features_financials.csv",
}
BRONZE = {
    "lms": "datamart/bronze/lms_loan_daily/",
    "clickstream": "datamart/bronze/feature_clickstream/",
    "attributes": "datamart/bronze/features_attributes/",
    "financials": "datamart/bronze/features_financials/",
}
SILVER = {
    "lms": "datamart/silver/lms_loan_daily/",
    "clickstream": "datamart/silver/features_clickstream/",
    "attributes": "datamart/silver/features_attributes/",
    "financials": "datamart/silver/features_financials/",
}
GOLD_LABEL_STORE = "datamart/gold/label_store/"
GOLD_FEATURE_STORE = "datamart/gold/feature_store/"

SKIP_EXIT_CODE = 99


def source_row_count(spark, source, date_str):
    snapshot_date = datetime.strptime(date_str, "%Y-%m-%d")
    return spark.read.csv(SOURCE[source], header=True, inferSchema=True) \
        .filter(col("snapshot_date") == snapshot_date).count()


def run_labels(spark, date_str):
    if source_row_count(spark, "lms", date_str) == 0:
        print(f"no loan data for {date_str}")
        return False
    utils.data_processing_bronze.process_bronze_table(date_str, SOURCE["lms"], BRONZE["lms"], spark)
    utils.data_processing_silver.process_lms(date_str, BRONZE["lms"], SILVER["lms"], spark)
    utils.data_processing_gold.process_label_store(date_str, SILVER["lms"], GOLD_LABEL_STORE, spark)
    return True


def run_features(spark, date_str):
    # attributes + financials define the applicant population of the month
    if source_row_count(spark, "attributes", date_str) == 0:
        print(f"no applications for {date_str}")
        return False

    for source in ["attributes", "financials", "clickstream"]:
        utils.data_processing_bronze.process_bronze_table(date_str, SOURCE[source], BRONZE[source], spark)

    utils.data_processing_silver.process_attributes(date_str, BRONZE["attributes"], SILVER["attributes"], spark)
    utils.data_processing_silver.process_financials(date_str, BRONZE["financials"], SILVER["financials"], spark)
    # clickstream ends before the last application months - gold fills fe_* with nulls when silver is absent
    if source_row_count(spark, "clickstream", date_str) > 0:
        utils.data_processing_silver.process_clickstream(date_str, BRONZE["clickstream"], SILVER["clickstream"], spark)
    else:
        print(f"no clickstream for {date_str} - fe_* will be null in the feature store")

    utils.data_processing_gold.process_feature_store(date_str, SILVER["attributes"], SILVER["financials"],
                                                     SILVER["clickstream"], GOLD_FEATURE_STORE, spark)
    return True


def main(snapshotdate, stage, spark=None):
    for d in list(BRONZE.values()) + list(SILVER.values()) + [GOLD_LABEL_STORE, GOLD_FEATURE_STORE]:
        os.makedirs(d, exist_ok=True)

    own_spark = spark is None
    if own_spark:
        spark = pyspark.sql.SparkSession.builder.appName(f"data_pipeline_{stage}").master("local[*]").getOrCreate()
        spark.sparkContext.setLogLevel("ERROR")
    try:
        return run_labels(spark, snapshotdate) if stage == "labels" else run_features(spark, snapshotdate)
    finally:
        if own_spark:
            spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run data pipeline for one month")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--stage", choices=["labels", "features"], required=True)
    args = parser.parse_args()

    if not main(args.snapshotdate, args.stage):
        sys.exit(SKIP_EXIT_CODE)
