"""
Train the credit default model for one training date and save the artefact to model_bank/.

Usage (from the project root):
    python model_train.py --model_train_date 2024-09-01
    python model_train.py --model_train_date 2025-04-01 --train_test_months 12 --oot_months 2
"""

import argparse
import pprint

import pyspark

import utils.model_train as mt


def main():
    parser = argparse.ArgumentParser(description="Train credit default model")
    parser.add_argument("--model_train_date", required=True, help="YYYY-MM-DD; only labels known before this date are used")
    parser.add_argument("--train_test_months", type=int, default=12)
    parser.add_argument("--oot_months", type=int, default=2)
    parser.add_argument("--train_test_ratio", type=float, default=0.8)
    parser.add_argument("--n_iter", type=int, default=20)
    parser.add_argument("--cv", type=int, default=3)
    parser.add_argument("--model_bank_dir", default=mt.MODEL_BANK_DIR)
    args = parser.parse_args()

    spark = pyspark.sql.SparkSession.builder.appName("model_train").master("local[*]").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    try:
        artefact = mt.run_training(
            spark,
            args.model_train_date,
            train_test_months=args.train_test_months,
            oot_months=args.oot_months,
            train_test_ratio=args.train_test_ratio,
            n_iter=args.n_iter,
            cv=args.cv,
            model_bank_dir=args.model_bank_dir,
        )
        print("\nmodel_version:", artefact["model_version"], "| type:", artefact["model_type"])
        pprint.pprint(artefact["results"])
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
