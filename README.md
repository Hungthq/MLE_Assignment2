# MLE Assignment 2: Credit Default ML Pipeline

An end-to-end machine learning pipeline that predicts, at the point of loan application, whether a customer will default. It runs monthly in Apache Airflow on Docker: it builds the data, trains a model, scores applications and monitors the model over time.

**Label:** `30dpd_6mob`, meaning the loan is 30 or more days past due at 6 months on book.

## Quick start

You need Docker Desktop with Docker Compose. The four source CSV files are already in `data/`.

```bash
git clone https://github.com/Hungthq/MLE_Assignment2.git
cd MLE_Assignment2
docker-compose build
docker-compose up
```

| Service | URL | Login |
|---|---|---|
| Airflow | http://localhost:8080 | `admin` / `admin` |
| JupyterLab | http://localhost:8822 | None |

These logins are for local use only.

The DAG starts paused. To run it, turn on `credit_model_pipeline` in the Airflow UI, or run this in a second terminal:

```bash
docker-compose exec airflow-scheduler airflow dags unpause credit_model_pipeline
```

Airflow then backfills 31 monthly runs, from 2023-01-01 to 2025-07-01, one month at a time.

To stop the services:

```bash
docker-compose down      # keeps the Airflow database
docker-compose down -v   # also deletes the Airflow database and run history
```

## What the DAG does

```text
start ─┬─ label_pipeline ─┬─ retrain_gate ── model_train ─┐
       │                  │                               ├─ model_inference ─┐
       └─ feature_pipeline ───────────────────────────────┘                   ├─ model_monitoring ── end
                          └───────────────────────────────────────────────────┘
```

Each run uses its logical date (`{{ ds }}`) as the snapshot month.

| Task | What it does | Script | When it does nothing |
|---|---|---|---|
| `label_pipeline` | Takes loans through bronze, silver and the gold label store | `utils/airflow_data_pipeline.py --stage labels` | No loan rows for the month |
| `feature_pipeline` | Takes attributes, financials and clickstream through bronze and silver to the gold feature store | `utils/airflow_data_pipeline.py --stage features` | No applications for the month (after 2025-01) |
| `retrain_gate` | Lets training run only on dates in `RETRAIN_DATES` | `dag/dag.py` | Skips `model_train` on all other dates |
| `model_train` | Trains and selects a model, then saves it to `model_bank/` | `utils/airflow_model_train.py` | Not a retrain date |
| `model_inference` | Scores the month's applicants with the newest model trained on or before that date | `utils/airflow_model_inference.py --modelname latest` | No model or no feature partition yet |
| `model_monitoring` | Calculates stability and, once labels mature, performance for each model | `utils/airflow_model_monitoring.py` | No predictions yet |

A script that exits with code 99 has nothing to do that month, and Airflow marks its task as skipped.

## Repository layout

```text
dag/dag.py                         Airflow DAG
utils/airflow_data_pipeline.py     Data pipeline for one month (Airflow task)
utils/airflow_model_train.py       Training script (Airflow task)
utils/airflow_model_inference.py   Scoring script (Airflow task)
utils/airflow_model_monitoring.py  Monitoring script (Airflow task)
utils/data_pipeline_all_months.py  Rebuilds all months in one run (manual use only)
utils/                             Bronze, silver, gold, training and monitoring logic
data/                              Source CSV files
datamart/                          Bronze, silver and gold outputs used by the pipeline
_datamart/                         Older copy of the outputs, not used by the pipeline
model_bank/                        Saved models (.pkl) and their metadata (.json)
*.ipynb                            Notebooks for EDA and development
Dockerfile.airflow                 Airflow image
Dockerfile.jupyter                 JupyterLab image
docker-compose.yaml                Airflow and JupyterLab services
requirements.txt                   Python dependencies
```

Inside the Airflow containers, the project folder is mounted at `/opt/airflow/scripts`, and the DAG runs each script from there.

## Outputs

| Output | Location |
|---|---|
| Bronze tables | `datamart/bronze/<table>/` |
| Silver tables | `datamart/silver/<table>/` |
| Feature store, one row per application | `datamart/gold/feature_store/` |
| Label store | `datamart/gold/label_store/` |
| Models and metadata | `model_bank/credit_model_<train date>.pkl` and `.json` |
| Predictions | `datamart/gold/model_predictions/<model>/` |
| Monitoring tables and charts | `datamart/gold/model_monitoring/<model>/` |

## Design rules

### Timing and leakage
- **Label date:** in the label store, `snapshot_date` is when the label becomes known, which is 6 months after the loan starts.
- **Feature date:** features are joined on the application date (`loan_start_date` equals the feature store's `snapshot_date`).
- **Training cut-off:** a model trained on date T uses only labels known before T. Training stops with an error if labels have no matching features, or if any feature is dated on or after T.
- **Clickstream:** only clickstream from the same month as the application is used. This assumes that month's snapshot is available at application.
- **Personal data:** `Name` and `SSN` are dropped before the gold feature store.

### Training and selection
With the default windows, the model trained on 2024-09-01 uses:

| Split | Label months | Purpose |
|---|---|---|
| Train and test (random 80/20, stratified) | 2023-07 to 2024-06 | Tuning, fitting and model selection |
| Out-of-time (OOT) | 2024-07 to 2024-08 | Final check and monitoring baseline |

- **Candidates:** logistic regression and XGBoost, each tuned with `RandomizedSearchCV` (20 draws, 3-fold CV, ROC AUC).
- **Selection:** the candidate with the highest test AUC wins. OOT results are reported but not used to choose.
- **Saved model:** each `.pkl` holds the full scikit-learn pipeline, feature list, training dates, metrics and the baseline distributions used for monitoring.

### Monitoring
Each run writes one table per model, with one row per scored application month.

| Check | Available | Green | Amber | Red |
|---|---|---|---|---|
| Score PSI against the training baseline | When the month is scored | ≤ 0.10 | ≤ 0.25 | > 0.25 |
| Highest feature PSI | When the month is scored | ≤ 0.10 | ≤ 0.25 | > 0.25 |
| Relative Gini drop against the OOT Gini | Once labels mature, 6 months later | ≤ 10% | ≤ 20% | > 20% |

A month's overall status is the worst of the three. Gini is reported only when a month has at least 50 labelled rows and both label classes.

## Current results

These results come from a full Airflow backfill. The DAG trained `credit_model_2024_09_01`, an XGBoost model with 57 input features:

| Split | AUC | Gini |
|---|---|---|
| Train | 0.917 | 0.834 |
| Test | 0.844 | 0.688 |
| OOT | 0.826 | 0.653 |

Monitoring as of 2025-07-01:

| Application month | Clickstream missing | Score PSI | Gini | Gini status | Mean score | Bad rate | Overall status |
|---|---|---|---|---|---|---|---|
| 2024-09 | 100% | 0.949 | 0.602 | GREEN | 0.609 | 0.331 | RED |
| 2024-10 | 100% | 1.051 | 0.622 | GREEN | 0.597 | 0.246 | RED |
| 2024-11 | 100% | 1.063 | 0.603 | GREEN | 0.618 | 0.281 | RED |
| 2024-12 | 100% | 0.685 | 0.700 | GREEN | 0.572 | 0.272 | RED |
| 2025-01 | 100% | 1.025 | 0.565 | AMBER | 0.610 | 0.268 | RED |

The model still ranks customers close to its baseline. Every month is RED because of PSI, which is driven by the missing clickstream features.

## Known limitations

- **Clickstream gap:** applicants from 2024-07 onward are not in the clickstream file, so every month this model has scored has no clickstream at all. Retraining alone will not fix it, because the fix has to come from the data source or from a model that doesn't use these features.
- **Scores are not calibrated:** the mean score is about 0.60, while the actual bad rate is about 0.28. Use the scores as a ranking, not as default probabilities, unless they are calibrated first.
- **Skipped scoring can hide a failure:** if training fails, later inference runs skip with "no model" instead of failing.
- **No automated response:** monitoring records a status, but it doesn't send alerts or start retraining. A new model is used automatically once it exists, with no comparison against the previous one.
- **Local setup only:** the project uses SQLite, `SequentialExecutor`, a file-based model bank and fixed local logins. Production would need a managed database, a parallel executor, a model registry and a secrets store.

## Run one step manually

Run these from the project folder while the services are up. Replace the dates as needed. The scripts live in `utils/` and run as modules (`python3 -m utils.<script>`) from the project root, so their `utils.` imports and the relative `data/`, `datamart/` and `model_bank/` paths keep working.

```bash
docker-compose exec airflow-scheduler bash -c "cd /opt/airflow/scripts && python3 -m utils.airflow_data_pipeline --snapshotdate 2024-09-01 --stage labels"
docker-compose exec airflow-scheduler bash -c "cd /opt/airflow/scripts && python3 -m utils.airflow_data_pipeline --snapshotdate 2024-09-01 --stage features"
docker-compose exec airflow-scheduler bash -c "cd /opt/airflow/scripts && python3 -m utils.airflow_model_train --model_train_date 2024-09-01"
docker-compose exec airflow-scheduler bash -c "cd /opt/airflow/scripts && python3 -m utils.airflow_model_inference --snapshotdate 2024-10-01 --modelname latest"
docker-compose exec airflow-scheduler bash -c "cd /opt/airflow/scripts && python3 -m utils.airflow_model_monitoring --snapshotdate 2025-07-01"
```

To add a retraining date, add it to `RETRAIN_DATES` in `dag/dag.py`. The date needs enough mature labels for the 12-month training and 2-month OOT windows.

## Troubleshooting

| Problem | What to do |
|---|---|
| The DAG doesn't appear | Run `docker-compose exec airflow-scheduler airflow dags list-import-errors` |
| Port 8080 or 8822 is in use | Stop the other service, or change the host port in `docker-compose.yaml` |
| A task runs out of memory | Give Docker Desktop more memory. Spark runs locally inside the task |
| A task fails with a permission error on Linux | Make the project folder writable by the container's `airflow` user |
| You want to rerun a month | Clear that run's tasks in the Airflow UI. Outputs for that month are overwritten |
