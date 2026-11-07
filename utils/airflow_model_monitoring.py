"""
Monitor every model that has predictions in datamart/gold/model_predictions/, as of a snapshot date.

Usage (from the project root):
    python -m utils.airflow_model_monitoring --snapshotdate 2025-07-01

Writes datamart/gold/model_monitoring/<model>/<model>_monitoring_<date>.parquet (+ .png chart).
Exit code 99 = nothing to monitor yet (Airflow marks the task as skipped).
"""

import argparse
import sys

import pyspark

import utils.model_monitoring as mm


def main(snapshotdate, spark=None):
    print("\n\n---starting job---\n\n")
    own_spark = spark is None
    if own_spark:
        spark = pyspark.sql.SparkSession.builder.appName("model_monitoring").master("local[*]").getOrCreate()
        spark.sparkContext.setLogLevel("ERROR")
    try:
        results = mm.run_monitoring(spark, snapshotdate)
    finally:
        if own_spark:
            spark.stop()
    print("\n\n---completed job---\n\n")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run model monitoring")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD (as-of date)")
    args = parser.parse_args()

    results = main(args.snapshotdate)
    if not results:
        print("no model predictions to monitor yet")
        sys.exit(99)
