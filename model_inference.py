import argparse
import os
import glob
import pickle
import pprint
import sys
from datetime import datetime

import pandas as pd
import pyspark
import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import DecimalType


# to call this script: python model_inference.py --snapshotdate "2024-09-01" --modelname "credit_model_2024_09_01.pkl"
#
# snapshotdate = application month in the gold feature store (datamart/gold/feature_store/).
# The model artefact is a full sklearn Pipeline (preprocessing + model) trained by model_train.py,
# so it takes the raw gold feature columns listed in model_artefact["feature_cols"].

FEATURE_STORE_DIR = "datamart/gold/feature_store/"
MODEL_BANK_DIR = "model_bank/"
PREDICTIONS_DIR = "datamart/gold/model_predictions/"


SKIP_EXIT_CODE = 99   # Airflow BashOperator marks the task as skipped on this exit code


def resolve_latest_model(snapshotdate, model_bank_dir=MODEL_BANK_DIR):
    """Newest artefact trained on or before snapshotdate (credit_model_YYYY_MM_DD.pkl), or None."""
    snapshot = datetime.strptime(snapshotdate, "%Y-%m-%d")
    candidates = []
    for path in glob.glob(os.path.join(model_bank_dir, "credit_model_*.pkl")):
        name = os.path.basename(path)
        try:
            train_date = datetime.strptime(name[len("credit_model_"):-4], "%Y_%m_%d")
        except ValueError:
            continue
        if train_date <= snapshot:
            candidates.append((train_date, name))
    return max(candidates)[1] if candidates else None


def feature_partition_exists(snapshotdate, feature_store_dir=FEATURE_STORE_DIR):
    partition = "gold_feature_store_" + snapshotdate.replace("-", "_") + ".parquet"
    return os.path.exists(os.path.join(feature_store_dir, partition))


def build_config(snapshotdate, modelname):
    config = {}
    config["snapshot_date_str"] = snapshotdate
    config["snapshot_date"] = datetime.strptime(config["snapshot_date_str"], "%Y-%m-%d")
    config["model_name"] = modelname
    config["model_bank_directory"] = MODEL_BANK_DIR
    config["model_artefact_filepath"] = config["model_bank_directory"] + config["model_name"]
    config["feature_store_directory"] = FEATURE_STORE_DIR
    return config


def load_model_artefact(config):
    with open(config["model_artefact_filepath"], "rb") as file:
        model_artefact = pickle.load(file)
    print("Model loaded successfully! " + config["model_artefact_filepath"])
    return model_artefact


def load_features(spark, config):
    """Read the gold feature store partition for one application month."""
    partition = "gold_feature_store_" + config["snapshot_date_str"].replace("-", "_") + ".parquet"
    filepath = os.path.join(config["feature_store_directory"], partition)
    if not os.path.exists(filepath):
        available = sorted(os.path.basename(p) for p in glob.glob(os.path.join(config["feature_store_directory"], "*.parquet")))
        raise FileNotFoundError(f"{filepath} not found. Available partitions: {available[0]} .. {available[-1]}")

    features_sdf = spark.read.parquet(filepath)
    # Decimal -> double, same as training (pandas would otherwise get decimal.Decimal objects)
    for field in features_sdf.schema.fields:
        if isinstance(field.dataType, DecimalType):
            features_sdf = features_sdf.withColumn(field.name, col(field.name).cast("double"))

    features_sdf = features_sdf.filter(col("snapshot_date") == config["snapshot_date"])
    print("extracted features_sdf", features_sdf.count(), config["snapshot_date"], "from", filepath)
    return features_sdf.toPandas()


def predict(model_artefact, features_pdf, config):
    """Score with the artefact's pipeline; returns the predictions table."""
    feature_cols = model_artefact["feature_cols"]
    missing = [c for c in feature_cols if c not in features_pdf.columns]
    if missing:
        raise KeyError(f"feature store is missing columns the model needs: {missing}")

    X_inference = features_pdf[feature_cols]
    print("X_inference", X_inference.shape,
          "| clickstream missing:", round(float(X_inference.filter(like="fe_").isna().all(axis=1).mean()), 3))

    y_inference = model_artefact["model"].predict_proba(X_inference)[:, 1]

    y_inference_pdf = features_pdf[["Customer_ID", "snapshot_date"]].copy()
    y_inference_pdf["model_name"] = config["model_name"]
    y_inference_pdf["model_predictions"] = y_inference
    return y_inference_pdf


def save_predictions(spark, y_inference_pdf, config):
    model_version = config["model_name"][:-4]
    gold_directory = os.path.join(PREDICTIONS_DIR, model_version) + "/"
    os.makedirs(gold_directory, exist_ok=True)

    # save gold table - IRL connect to database to write
    partition_name = model_version + "_predictions_" + config["snapshot_date_str"].replace("-", "_") + ".parquet"
    filepath = gold_directory + partition_name
    spark.createDataFrame(y_inference_pdf).write.mode("overwrite").parquet(filepath)
    print("saved to:", filepath)
    return filepath


def main(snapshotdate, modelname, spark=None):
    print("\n\n---starting job---\n\n")

    # reuse the caller's session (e.g. a notebook) and only stop a session this job created
    own_spark = spark is None
    if own_spark:
        spark = pyspark.sql.SparkSession.builder.appName("model_inference").master("local[*]").getOrCreate()
        spark.sparkContext.setLogLevel("ERROR")

    try:
        config = build_config(snapshotdate, modelname)
        pprint.pprint(config)

        model_artefact = load_model_artefact(config)
        features_pdf = load_features(spark, config)
        y_inference_pdf = predict(model_artefact, features_pdf, config)
        filepath = save_predictions(spark, y_inference_pdf, config)
    finally:
        if own_spark:
            spark.stop()

    print("\n\n---completed job---\n\n")
    return filepath


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD (application month in gold feature store)")
    parser.add_argument("--modelname", type=str, required=True,
                        help="model_name, e.g. credit_model_2024_09_01.pkl, or 'latest' = newest model trained on/before snapshotdate")
    args = parser.parse_args()

    modelname = args.modelname
    if modelname == "latest":
        modelname = resolve_latest_model(args.snapshotdate)
        if modelname is None:
            print(f"no model trained on or before {args.snapshotdate} - nothing to score")
            sys.exit(SKIP_EXIT_CODE)
        print("latest model:", modelname)

    if not feature_partition_exists(args.snapshotdate):
        print(f"no gold feature store partition for {args.snapshotdate} - nothing to score")
        sys.exit(SKIP_EXIT_CODE)

    main(args.snapshotdate, modelname)
