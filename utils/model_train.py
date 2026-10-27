"""
Model training utilities for the credit default model (label: 30dpd_6mob).

Shared by `model_train.py` (CLI) and `model_train_main.ipynb` (notebook) so the two
can never drift apart.

Point-in-time rules
-------------------
* The label store's `snapshot_date` is the date the label becomes KNOWN
  (loan_start_date + 6 months on book). Train / test / OOT windows are cut on it,
  so a model trained on date T only uses labels that existed before T.
* Features are joined on the APPLICATION date:
  label.loan_start_date == feature_store.snapshot_date
"""

import os
import glob
import json
import pickle
import pprint
from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd

import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import DecimalType

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import xgboost as xgb


# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
FEATURE_STORE_DIR = "datamart/gold/feature_store/"
LABEL_STORE_DIR = "datamart/gold/label_store/"
MODEL_BANK_DIR = "model_bank/"

SEED = 88

# never used as model inputs
ID_DATE_COLS = ["Customer_ID", "snapshot_date", "loan_id", "loan_start_date",
                "label", "label_def", "label_snapshot_date", "feature_snapshot_date"]
# redundant: already split into Spend_Level / Payment_Value and Credit_Mix + Payment_of_Min_Amount
REDUNDANT_COLS = ["Payment_Behaviour", "Credit_Mix_MinAmt"]
CLICKSTREAM_PREFIX = "fe_"


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
def build_config(model_train_date_str, train_test_months=12, oot_months=2, train_test_ratio=0.8):
    """Date windows for one training run. All windows are on the LABEL snapshot_date."""
    config = {}
    config["model_train_date_str"] = model_train_date_str
    config["train_test_period_months"] = train_test_months
    config["oot_period_months"] = oot_months
    config["model_train_date"] = datetime.strptime(model_train_date_str, "%Y-%m-%d")
    config["oot_end_date"] = config["model_train_date"] - timedelta(days=1)
    config["oot_start_date"] = config["model_train_date"] - relativedelta(months=oot_months)
    config["train_test_end_date"] = config["oot_start_date"] - timedelta(days=1)
    config["train_test_start_date"] = config["oot_start_date"] - relativedelta(months=train_test_months)
    config["train_test_ratio"] = train_test_ratio
    return config


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------
def read_store(spark, folder_path):
    """Read every monthly parquet partition in a gold store folder."""
    files_list = sorted(glob.glob(os.path.join(folder_path, "*.parquet")))
    if not files_list:
        raise FileNotFoundError(f"no parquet partitions found in {folder_path}")
    return spark.read.parquet(*files_list)


def load_training_data(spark, config, feature_store_dir=FEATURE_STORE_DIR, label_store_dir=LABEL_STORE_DIR,
                       verbose=True):
    """
    Labels known in [train_test_start_date, oot_end_date] joined to the features captured
    at application. Returns a pandas DataFrame with `label_snapshot_date` (split key)
    and `feature_snapshot_date` (= loan_start_date).
    """
    label_store_sdf = read_store(spark, label_store_dir)
    labels_sdf = label_store_sdf.filter(
        (col("snapshot_date") >= config["train_test_start_date"]) &
        (col("snapshot_date") <= config["oot_end_date"])
    ).withColumnRenamed("snapshot_date", "label_snapshot_date")

    features_sdf = read_store(spark, feature_store_dir)
    # Decimal -> double so pandas gets floats rather than decimal.Decimal objects
    for field in features_sdf.schema.fields:
        if isinstance(field.dataType, DecimalType):
            features_sdf = features_sdf.withColumn(field.name, col(field.name).cast("double"))
    features_sdf = features_sdf.withColumnRenamed("snapshot_date", "feature_snapshot_date")

    data_sdf = labels_sdf.join(
        features_sdf,
        on=[labels_sdf["Customer_ID"] == features_sdf["Customer_ID"],
            labels_sdf["loan_start_date"] == features_sdf["feature_snapshot_date"]],
        how="left",
    ).drop(features_sdf["Customer_ID"])

    data_pdf = data_sdf.toPandas()

    # window guard: fail early with a readable message instead of an empty train set later
    if len(data_pdf) == 0:
        rng = label_store_sdf.agg(F.min("snapshot_date"), F.max("snapshot_date")).first()
        raise ValueError(
            f"no labels between {config['train_test_start_date'].date()} and {config['oot_end_date'].date()}. "
            f"Label store covers {rng[0]} -> {rng[1]}; pick a model_train_date so the "
            f"{config['train_test_period_months']}+{config['oot_period_months']} month window falls inside it."
        )

    # point-in-time guard: features must be from before the training date
    assert (pd.to_datetime(data_pdf["feature_snapshot_date"]) < config["model_train_date"]).all(), \
        "feature snapshot after model_train_date - leakage"
    n_unmatched = data_pdf["feature_snapshot_date"].isna().sum()

    if verbose:
        print("labels in window:", len(data_pdf), "| labels without features:", n_unmatched,
              "|", config["train_test_start_date"].date(), "->", config["oot_end_date"].date())
    return data_pdf


# ---------------------------------------------------------------------------
# splitting
# ---------------------------------------------------------------------------
def get_feature_columns(data_pdf):
    """Model inputs split into numeric and categorical columns."""
    drop = set(ID_DATE_COLS + REDUNDANT_COLS)
    feature_cols = [c for c in data_pdf.columns if c not in drop]
    categorical_cols = [c for c in feature_cols if data_pdf[c].dtype == object]
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]
    return feature_cols, numeric_cols, categorical_cols


def split_data(data_pdf, config, feature_cols, verbose=True):
    """Train / test (random, stratified) inside the train_test window; OOT = the later window."""
    snap = pd.to_datetime(data_pdf["label_snapshot_date"])
    oot_pdf = data_pdf[(snap >= config["oot_start_date"]) & (snap <= config["oot_end_date"])]
    train_test_pdf = data_pdf[(snap >= config["train_test_start_date"]) & (snap <= config["train_test_end_date"])]
    if len(train_test_pdf) == 0 or len(oot_pdf) == 0:
        raise ValueError(f"empty window: train_test rows={len(train_test_pdf)}, oot rows={len(oot_pdf)} "
                         f"for model_train_date {config['model_train_date_str']}")

    X_train, X_test, y_train, y_test = train_test_split(
        train_test_pdf[feature_cols], train_test_pdf["label"],
        test_size=1 - config["train_test_ratio"],
        random_state=SEED,
        shuffle=True,
        stratify=train_test_pdf["label"],
    )
    splits = {
        "train": (X_train, y_train),
        "test": (X_test, y_test),
        "oot": (oot_pdf[feature_cols], oot_pdf["label"]),
    }
    if verbose:
        for name, (X, y) in splits.items():
            fe_null = X[[c for c in X.columns if c.startswith(CLICKSTREAM_PREFIX)]].isna().all(axis=1).mean()
            print(f"{name:5s} rows={len(X):5d}  bad_rate={y.mean():.3f}  clickstream_missing={fe_null:.1%}")
    return splits


# ---------------------------------------------------------------------------
# preprocessing
# ---------------------------------------------------------------------------
def clean_categoricals(X):
    """'_' and nulls in categorical columns -> 'Unknown' (e.g. Credit_Mix == '_')."""
    X = pd.DataFrame(X).copy()
    for c in X.columns:
        X[c] = X[c].astype(object).where(X[c].notna(), "Unknown").replace("_", "Unknown")
    return X


def build_preprocessor(numeric_cols, categorical_cols, model_type):
    """
    One ColumnTransformer, fitted on train only, pickled with the model.

    logreg : median-impute + missing flag on ALL numerics (incl. clickstream), then scale
    xgboost: median-impute + missing flag on non-clickstream numerics; clickstream passes
             through with NaN so XGBoost's native missing-value handling is used; no scaling
    """
    cat_pipe = Pipeline([
        ("clean", FunctionTransformer(clean_categoricals, feature_names_out="one-to-one")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    if model_type == "logreg":
        num_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
            ("scale", StandardScaler()),
        ])
        transformers = [("num", num_pipe, numeric_cols)]
    elif model_type == "xgboost":
        click_cols = [c for c in numeric_cols if c.startswith(CLICKSTREAM_PREFIX)]
        other_cols = [c for c in numeric_cols if c not in click_cols]
        transformers = [
            ("num", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True), other_cols),
            ("click", "passthrough", click_cols),
        ]
    else:
        raise ValueError(model_type)

    transformers.append(("cat", cat_pipe, categorical_cols))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
def candidate_searches(numeric_cols, categorical_cols, y_train, n_iter=20, cv=3):
    """RandomizedSearchCV for each candidate model (scored on predicted probabilities)."""
    neg_pos = float((y_train == 0).sum() / (y_train == 1).sum())

    logreg = Pipeline([
        ("preprocess", build_preprocessor(numeric_cols, categorical_cols, "logreg")),
        ("model", LogisticRegression(max_iter=2000, random_state=SEED)),
    ])
    logreg_params = {
        "model__C": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10],
        "model__class_weight": [None, "balanced"],
    }

    xgb_pipe = Pipeline([
        ("preprocess", build_preprocessor(numeric_cols, categorical_cols, "xgboost")),
        ("model", xgb.XGBClassifier(eval_metric="logloss", random_state=SEED, n_jobs=1)),
    ])
    xgb_params = {
        "model__n_estimators": [50, 100, 200, 300],
        "model__max_depth": [2, 3, 4, 5],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "model__subsample": [0.6, 0.8, 1.0],
        "model__colsample_bytree": [0.6, 0.8, 1.0],
        "model__gamma": [0, 0.1, 0.5],
        "model__min_child_weight": [1, 3, 5, 10],
        "model__reg_alpha": [0, 0.1, 1],
        "model__reg_lambda": [1, 1.5, 2, 5],
        "model__scale_pos_weight": [1, round(neg_pos, 2)],
    }

    searches = {}
    for name, (pipe, params) in {"logreg": (logreg, logreg_params), "xgboost": (xgb_pipe, xgb_params)}.items():
        n_combos = int(np.prod([len(v) for v in params.values()]))
        searches[name] = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=params,
            scoring="roc_auc",          # uses predict_proba (fixes make_scorer(roc_auc_score) bug)
            n_iter=min(n_iter, n_combos),
            cv=cv,
            random_state=42,
            n_jobs=-1,
            verbose=0,
        )
    return searches


def evaluate(model, splits):
    """AUC and Gini on every split."""
    results = {}
    for name, (X, y) in splits.items():
        auc = roc_auc_score(y, model.predict_proba(X)[:, 1])
        results[f"auc_{name}"] = round(float(auc), 4)
        results[f"gini_{name}"] = round(float(2 * auc - 1), 3)
    return results


def train_and_select(splits, numeric_cols, categorical_cols, n_iter=20, cv=3, verbose=True):
    """Tune every candidate on train, select the winner on TEST AUC (OOT is report-only)."""
    X_train, y_train = splits["train"]
    searches = candidate_searches(numeric_cols, categorical_cols, y_train, n_iter=n_iter, cv=cv)

    candidates = {}
    for name, search in searches.items():
        search.fit(X_train, y_train)
        res = evaluate(search.best_estimator_, splits)
        res["cv_auc"] = round(float(search.best_score_), 4)
        candidates[name] = {
            "model": search.best_estimator_,
            "best_params": {k.replace("model__", ""): v for k, v in search.best_params_.items()},
            "results": res,
        }
        if verbose:
            print(f"{name:8s} cv_auc={res['cv_auc']:.4f}  gini train/test/oot = "
                  f"{res['gini_train']:.3f} / {res['gini_test']:.3f} / {res['gini_oot']:.3f}")

    best_name = max(candidates, key=lambda n: candidates[n]["results"]["auc_test"])
    if verbose:
        print("selected (highest test AUC):", best_name)
    return best_name, candidates


# ---------------------------------------------------------------------------
# artefact
# ---------------------------------------------------------------------------
def build_artefact(config, best_name, candidates, splits, feature_cols, numeric_cols, categorical_cols, label_def):
    best = candidates[best_name]
    model_artefact = {}
    model_artefact["model"] = best["model"]               # full sklearn Pipeline: preprocess + model
    model_artefact["model_type"] = best_name
    model_artefact["model_version"] = "credit_model_" + config["model_train_date_str"].replace("-", "_")
    model_artefact["label_def"] = label_def
    model_artefact["feature_cols"] = feature_cols
    model_artefact["numeric_cols"] = numeric_cols
    model_artefact["categorical_cols"] = categorical_cols
    model_artefact["data_dates"] = config
    model_artefact["data_stats"] = {}
    for name, (X, y) in splits.items():
        model_artefact["data_stats"][f"X_{name}"] = int(X.shape[0])
        model_artefact["data_stats"][f"y_{name}"] = round(float(y.mean()), 3)
        model_artefact["data_stats"][f"clickstream_missing_{name}"] = round(float(
            X[[c for c in X.columns if c.startswith(CLICKSTREAM_PREFIX)]].isna().all(axis=1).mean()), 3)
    model_artefact["results"] = best["results"]
    model_artefact["hp_params"] = best["best_params"]
    model_artefact["candidates"] = {n: {"results": c["results"], "hp_params": c["best_params"]}
                                    for n, c in candidates.items()}
    return model_artefact


def _jsonable(obj):
    if isinstance(obj, (datetime, date)):
        return obj.strftime("%Y-%m-%d")
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return str(obj)


def save_artefact(model_artefact, model_bank_dir=MODEL_BANK_DIR, verbose=True):
    """Pickle the full artefact + a JSON sidecar (everything except the model object)."""
    os.makedirs(model_bank_dir, exist_ok=True)
    pkl_path = os.path.join(model_bank_dir, model_artefact["model_version"] + ".pkl")
    json_path = os.path.join(model_bank_dir, model_artefact["model_version"] + ".json")

    with open(pkl_path, "wb") as f:
        pickle.dump(model_artefact, f)

    meta = {k: v for k, v in model_artefact.items() if k != "model"}
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2, default=_jsonable)

    if verbose:
        print("model saved to", pkl_path)
        print("metrics saved to", json_path)
    return pkl_path, json_path


def load_artefact(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# end-to-end
# ---------------------------------------------------------------------------
def run_training(spark, model_train_date_str, train_test_months=12, oot_months=2, train_test_ratio=0.8,
                 n_iter=20, cv=3, feature_store_dir=FEATURE_STORE_DIR, label_store_dir=LABEL_STORE_DIR,
                 model_bank_dir=MODEL_BANK_DIR, verbose=True):
    config = build_config(model_train_date_str, train_test_months, oot_months, train_test_ratio)
    if verbose:
        pprint.pprint(config)

    data_pdf = load_training_data(spark, config, feature_store_dir, label_store_dir, verbose=verbose)
    feature_cols, numeric_cols, categorical_cols = get_feature_columns(data_pdf)
    splits = split_data(data_pdf, config, feature_cols, verbose=verbose)

    best_name, candidates = train_and_select(splits, numeric_cols, categorical_cols, n_iter=n_iter, cv=cv,
                                             verbose=verbose)
    label_def = data_pdf["label_def"].iloc[0]
    model_artefact = build_artefact(config, best_name, candidates, splits, feature_cols, numeric_cols,
                                    categorical_cols, label_def)
    save_artefact(model_artefact, model_bank_dir, verbose=verbose)
    return model_artefact
