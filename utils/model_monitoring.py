"""
Model monitoring for the credit default model (label: 30dpd_6mob).

For one as-of date and one model, builds a table with one row per APPLICATION month the model scored:

* stability  - available as soon as the month is scored
    score_psi        PSI of the score distribution vs the training baseline (held-out test split)
    max_feature_psi  largest PSI over the model's input features (+ which feature)
    clickstream_missing  share of applicants with no clickstream at all
* performance - available only once labels mature (application + 6 months on book) AND are known by as_of
    bad_rate, auc, gini on the labelled rows
* period     - where the month sits relative to the model's own training data
    "train_test" / "oot" (seen during development, optimistic) or "post_training" (true out-of-time)
* status     - GREEN / AMBER / RED from the thresholds below; overall = worst of the three checks

The baseline distributions are stored in the model artefact at training time
(`build_distribution_baseline`, called by utils.model_train.build_artefact).
"""

import os
import glob
from datetime import datetime
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd

from pyspark.sql.functions import col
from pyspark.sql.types import DecimalType

from sklearn.metrics import roc_auc_score


FEATURE_STORE_DIR = "datamart/gold/feature_store/"
LABEL_STORE_DIR = "datamart/gold/label_store/"
PREDICTIONS_DIR = "datamart/gold/model_predictions/"
MONITORING_DIR = "datamart/gold/model_monitoring/"
MODEL_BANK_DIR = "model_bank/"

LABEL_MOB_MONTHS = 6          # label is known 6 months after application
MIN_LABELLED_ROWS = 50        # below this, AUC/Gini are too noisy to report

# thresholds
PSI_AMBER, PSI_RED = 0.10, 0.25                  # industry rule of thumb
GINI_DROP_AMBER, GINI_DROP_RED = 0.10, 0.20      # relative drop vs the model's OOT Gini

EPS = 1e-4
N_BINS = 10
STATUS_RANK = {"GREEN": 0, "AMBER": 1, "RED": 2, "N/A": -1}


# ---------------------------------------------------------------------------
# distributions & PSI
# ---------------------------------------------------------------------------
def _clean_cat(s):
    s = pd.Series(s).astype(object)
    return s.where(s.notna(), "Unknown").replace("_", "Unknown").astype(str)


def numeric_spec(values, n_bins=N_BINS):
    """Quantile bin edges from the baseline + proportion per bin; NaN gets its own bin (last)."""
    v = pd.to_numeric(pd.Series(values), errors="coerce").astype(float)
    non_null = v.dropna()
    edges = np.unique(np.quantile(non_null, np.linspace(0, 1, n_bins + 1)[1:-1])) if len(non_null) else np.array([])
    return {"type": "numeric", "edges": edges.tolist(), "props": numeric_props(v, edges).tolist()}


def numeric_props(values, edges):
    v = pd.to_numeric(pd.Series(values), errors="coerce").astype(float)
    edges = np.asarray(edges, dtype=float)
    counts = np.zeros(len(edges) + 2)
    non_null = v.dropna().to_numpy()
    if len(non_null):
        idx = np.searchsorted(edges, non_null, side="right")
        counts[: len(edges) + 1] = np.bincount(idx, minlength=len(edges) + 1)
    counts[-1] = v.isna().sum()
    return counts / max(len(v), 1)


def categorical_spec(values):
    s = _clean_cat(values)
    cats = sorted(s.unique().tolist())
    return {"type": "categorical", "categories": cats, "props": categorical_props(s, cats).tolist()}


def categorical_props(values, categories):
    s = _clean_cat(values)
    counts = [float((s == c).sum()) for c in categories] + [float((~s.isin(categories)).sum())]  # last = unseen
    return np.asarray(counts) / max(len(s), 1)


def psi(expected, actual):
    e = np.clip(np.asarray(expected, dtype=float), EPS, None)
    a = np.clip(np.asarray(actual, dtype=float), EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def build_distribution_baseline(X, scores, numeric_cols, categorical_cols):
    """Reference distributions for drift monitoring, computed on held-out data at training time."""
    features = {}
    for c in numeric_cols:
        features[c] = numeric_spec(X[c])
    for c in categorical_cols:
        features[c] = categorical_spec(X[c])
    score = numeric_spec(scores)
    return {"n_rows": int(len(X)), "score": score, "features": features,
            "avg_score": float(np.mean(scores))}


def score_psi(baseline, scores):
    spec = baseline["score"]
    return psi(spec["props"], numeric_props(scores, spec["edges"]))


def feature_psis(baseline, X):
    out = {}
    for c, spec in baseline["features"].items():
        if c not in X.columns:
            continue
        if spec["type"] == "numeric":
            out[c] = psi(spec["props"], numeric_props(X[c], spec["edges"]))
        else:
            props = spec["props"] + [0.0] if len(spec["props"]) == len(spec["categories"]) else spec["props"]
            out[c] = psi(props, categorical_props(X[c], spec["categories"]))
    return out


# ---------------------------------------------------------------------------
# status rules
# ---------------------------------------------------------------------------
def psi_status(value):
    if value is None or np.isnan(value):
        return "N/A"
    return "RED" if value > PSI_RED else "AMBER" if value > PSI_AMBER else "GREEN"


def gini_status(gini, reference_gini):
    if gini is None or np.isnan(gini) or not reference_gini:
        return "N/A"
    drop = (reference_gini - gini) / reference_gini
    return "RED" if drop > GINI_DROP_RED else "AMBER" if drop > GINI_DROP_AMBER else "GREEN"


def worst(*statuses):
    s = [x for x in statuses if x != "N/A"]
    return max(s, key=STATUS_RANK.get) if s else "N/A"


# ---------------------------------------------------------------------------
# data access (spark)
# ---------------------------------------------------------------------------
def _read_parquets(spark, folder, pattern="*.parquet"):
    files = sorted(glob.glob(os.path.join(folder, pattern)))
    return spark.read.parquet(*files) if files else None


def _decimals_to_double(sdf):
    for f in sdf.schema.fields:
        if isinstance(f.dataType, DecimalType):
            sdf = sdf.withColumn(f.name, col(f.name).cast("double"))
    return sdf


def models_with_predictions(predictions_dir=PREDICTIONS_DIR):
    return sorted(os.path.basename(os.path.normpath(p)) for p in glob.glob(os.path.join(predictions_dir, "*/"))
                  if glob.glob(os.path.join(p, "*.parquet")))


def load_monitoring_inputs(spark, model_version, as_of, feature_cols,
                           predictions_dir=PREDICTIONS_DIR, label_store_dir=LABEL_STORE_DIR,
                           feature_store_dir=FEATURE_STORE_DIR):
    """Predictions scored up to as_of, joined to the features scored and the labels KNOWN at as_of."""
    preds = _read_parquets(spark, os.path.join(predictions_dir, model_version))
    if preds is None:
        return None
    preds = preds.filter(col("snapshot_date") <= as_of)

    features = _decimals_to_double(_read_parquets(spark, feature_store_dir))
    features = features.select("Customer_ID", "snapshot_date", *feature_cols)

    labels = _read_parquets(spark, label_store_dir)
    labels = labels.filter(col("snapshot_date") <= as_of) \
        .select("Customer_ID", col("loan_start_date").alias("snapshot_date"), "label")

    df = preds.join(features, on=["Customer_ID", "snapshot_date"], how="left") \
              .join(labels, on=["Customer_ID", "snapshot_date"], how="left")
    pdf = df.toPandas()
    pdf["snapshot_date"] = pd.to_datetime(pdf["snapshot_date"])
    return pdf


# ---------------------------------------------------------------------------
# monitoring table
# ---------------------------------------------------------------------------
def period_of(app_month, data_dates):
    label_month = app_month + relativedelta(months=LABEL_MOB_MONTHS)
    if label_month <= data_dates["train_test_end_date"]:
        return "train_test"
    if label_month <= data_dates["oot_end_date"]:
        return "oot"
    return "post_training"


def monitor_model(pdf, model_artefact, as_of):
    """One row per application month."""
    baseline = model_artefact.get("baseline")
    if baseline is None:
        raise KeyError(f"{model_artefact['model_version']} has no 'baseline' - retrain it with the current model_train.py")
    reference_gini = model_artefact["results"]["gini_oot"]
    click_cols = [c for c in model_artefact["feature_cols"] if c.startswith("fe_")]

    rows = []
    for app_month, g in pdf.groupby("snapshot_date"):
        scores = g["model_predictions"].to_numpy()
        f_psi = feature_psis(baseline, g)
        top_feature = max(f_psi, key=f_psi.get) if f_psi else None

        lab = g[g["label"].notna()]
        n_lab = len(lab)
        if n_lab >= MIN_LABELLED_ROWS and lab["label"].nunique() == 2:
            auc = roc_auc_score(lab["label"].astype(int), lab["model_predictions"])
            gini = 2 * auc - 1
        else:
            auc = gini = np.nan

        s_psi = score_psi(baseline, scores)
        row = {
            "model_version": model_artefact["model_version"],
            "as_of_date": as_of.date(),
            "snapshot_date": app_month.date(),
            "label_known_date": (app_month + relativedelta(months=LABEL_MOB_MONTHS)).date(),
            "period": period_of(app_month, model_artefact["data_dates"]),
            "n_scored": int(len(g)),
            "avg_score": float(np.mean(scores)),
            "clickstream_missing": float(g[click_cols].isna().all(axis=1).mean()) if click_cols else np.nan,
            "score_psi": s_psi,
            "max_feature_psi": f_psi[top_feature] if top_feature else np.nan,
            "max_psi_feature": top_feature,
            "n_labelled": int(n_lab),
            "bad_rate": float(lab["label"].mean()) if n_lab else np.nan,
            "auc": float(auc),
            "gini": float(gini),
            "reference_gini": float(reference_gini),
        }
        row["status_score_psi"] = psi_status(row["score_psi"])
        row["status_feature_psi"] = psi_status(row["max_feature_psi"])
        row["status_gini"] = gini_status(row["gini"], reference_gini)
        row["status"] = worst(row["status_score_psi"], row["status_feature_psi"], row["status_gini"])
        rows.append(row)

    return pd.DataFrame(rows).sort_values("snapshot_date").reset_index(drop=True)


def plot_monitoring(mon_pdf, filepath=None):
    import matplotlib
    if filepath:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # reindex to every month so unscored months show as gaps instead of bridged lines
    mon_pdf = mon_pdf.copy()
    mon_pdf["snapshot_date"] = pd.to_datetime(mon_pdf["snapshot_date"])
    full = pd.date_range(mon_pdf["snapshot_date"].min(), mon_pdf["snapshot_date"].max(), freq="MS")
    mon_pdf = mon_pdf.set_index("snapshot_date").reindex(full).rename_axis("snapshot_date").reset_index()
    mon_pdf[["model_version", "as_of_date", "reference_gini"]] = \
        mon_pdf[["model_version", "as_of_date", "reference_gini"]].ffill().bfill()

    x = mon_pdf["snapshot_date"]
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    model_version = mon_pdf["model_version"].iloc[0]
    as_of = mon_pdf["as_of_date"].iloc[0]

    # shade the months the model saw during development
    dev = mon_pdf["period"].isin(["train_test", "oot"]).to_numpy()
    for ax in axes:
        if dev.any():
            ax.axvspan(x[dev].min() - pd.Timedelta(days=15), x[dev].max() + pd.Timedelta(days=15),
                       color="grey", alpha=0.12, lw=0)

    ax = axes[0]
    ax.plot(x, mon_pdf["gini"], marker="o", label="Gini (labelled)")
    ref = mon_pdf["reference_gini"].iloc[0]
    ax.axhline(ref, ls="--", c="k", lw=1, label=f"OOT Gini at training ({ref:.3f})")
    ax.axhline(ref * (1 - GINI_DROP_RED), ls=":", c="r", lw=1, label=f"RED (-{GINI_DROP_RED:.0%})")
    ax.set_ylabel("Gini")
    ax.legend(fontsize=8, loc="lower left")

    ax = axes[1]
    ax.plot(x, mon_pdf["score_psi"], marker="o", label="score PSI")
    ax.plot(x, mon_pdf["max_feature_psi"], marker="s", label="max feature PSI")
    ax.axhline(PSI_AMBER, ls=":", c="orange", lw=1)
    ax.axhline(PSI_RED, ls=":", c="r", lw=1)
    ax.set_yscale("symlog", linthresh=0.1)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("PSI")
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[2]
    ax.plot(x, mon_pdf["avg_score"], marker="o", label="avg score")
    ax.plot(x, mon_pdf["bad_rate"], marker="o", label="actual bad rate")
    ax.plot(x, mon_pdf["clickstream_missing"], ls="--", label="clickstream missing")
    ax.set_ylabel("rate")
    ax.set_xlabel("application month (shaded = seen in development)")
    ax.legend(fontsize=8, loc="upper left")

    fig.suptitle(f"{model_version} - monitoring as of {as_of}")
    fig.tight_layout()
    if filepath:
        fig.savefig(filepath, dpi=110)
        plt.close(fig)
    return fig


def save_monitoring(spark, mon_pdf, monitoring_dir=MONITORING_DIR, with_chart=True):
    """Gold table: one parquet per (model, as_of) holding the full month-by-month view at that date."""
    model_version = mon_pdf["model_version"].iloc[0]
    as_of_str = str(mon_pdf["as_of_date"].iloc[0]).replace("-", "_")
    out_dir = os.path.join(monitoring_dir, model_version)
    os.makedirs(out_dir, exist_ok=True)

    filepath = os.path.join(out_dir, f"{model_version}_monitoring_{as_of_str}.parquet")
    sdf_pdf = mon_pdf.copy()
    for c in sdf_pdf.select_dtypes(include="float").columns:
        sdf_pdf[c] = sdf_pdf[c].astype(object).where(sdf_pdf[c].notna(), None)
    spark.createDataFrame(sdf_pdf).write.mode("overwrite").parquet(filepath)
    print("saved to:", filepath)

    chart_path = None
    if with_chart:
        chart_path = os.path.join(out_dir, f"{model_version}_monitoring_{as_of_str}.png")
        plot_monitoring(mon_pdf, chart_path)
        print("chart saved to:", chart_path)
    return filepath, chart_path


def run_monitoring(spark, as_of_str, model_bank_dir=MODEL_BANK_DIR, verbose=True):
    """Monitor every model that has predictions; returns {model_version: table}."""
    import pickle

    as_of = datetime.strptime(as_of_str, "%Y-%m-%d")
    results = {}
    for model_version in models_with_predictions():
        artefact_path = os.path.join(model_bank_dir, model_version + ".pkl")
        if not os.path.exists(artefact_path):
            print(f"skip {model_version}: no artefact in {model_bank_dir}")
            continue
        with open(artefact_path, "rb") as f:
            model_artefact = pickle.load(f)

        pdf = load_monitoring_inputs(spark, model_version, as_of, model_artefact["feature_cols"])
        if pdf is None or pdf.empty:
            print(f"skip {model_version}: no predictions up to {as_of_str}")
            continue
        mon_pdf = monitor_model(pdf, model_artefact, as_of)
        save_monitoring(spark, mon_pdf)
        results[model_version] = mon_pdf

        if verbose:
            latest = mon_pdf.iloc[-1]
            labelled = mon_pdf[mon_pdf["gini"].notna() & (mon_pdf["period"] == "post_training")]
            print(f"{model_version}: {len(mon_pdf)} months | latest scored month {latest['snapshot_date']} "
                  f"status={latest['status']} (score_psi={latest['score_psi']:.3f}, "
                  f"max_feature_psi={latest['max_feature_psi']:.3f} [{latest['max_psi_feature']}])")
            if len(labelled):
                last = labelled.iloc[-1]
                print(f"   latest post-training Gini: {last['gini']:.3f} for {last['snapshot_date']} "
                      f"vs reference {last['reference_gini']:.3f} -> {last['status_gini']}")
    return results
