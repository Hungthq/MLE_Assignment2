"""
Credit default model - end-to-end monthly pipeline.

One DAG run per month ({{ ds }} = first of the month), backfilled from 2023-01-01 to 2025-07-01:

    start ─┬─ label_pipeline ─── retrain_gate ── model_train ─┐
           │        │                                          ├─ model_inference ─┐
           └─ feature_pipeline ────────────────────────────────┘                   ├─ model_monitoring ── end
                    └──────────────────────────────────────────────────────────────┘ (label_pipeline also feeds monitoring)

* label_pipeline    loans  -> bronze -> silver -> gold label store (labels that became known this month)
* feature_pipeline  attributes + financials + clickstream -> gold feature store (applications this month);
                    skipped after 2025-01 (no more applications in the source)
* retrain_gate      only lets model_train run on the dates in RETRAIN_DATES
* model_train       trains on labels known before {{ ds }} and saves model_bank/credit_model_<ds>.pkl
* model_inference   scores this month's applicants with the newest model trained on/before {{ ds }};
                    skipped before the first model exists or when there are no applications
* model_monitoring  per scored month: score PSI, feature PSI, clickstream missing, and Gini once labels mature

Scripts run from /opt/airflow/scripts (the project root mounted in docker-compose.yaml).
A script exiting with code 99 means "nothing to do this month" and Airflow marks the task as skipped.
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import ShortCircuitOperator
from airflow.utils.trigger_rule import TriggerRule


SCRIPTS_DIR = "/opt/airflow/scripts"
RETRAIN_DATES = ["2024-09-01"]   # first date with 12 + 2 months of known labels; add dates to retrain


def is_retrain_date(ds, **_):
    print(f"{ds} retrain date? {ds in RETRAIN_DATES} (RETRAIN_DATES={RETRAIN_DATES})")
    return ds in RETRAIN_DATES


def script(cmd):
    return f"cd {SCRIPTS_DIR} && {cmd}"


default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="credit_model_pipeline",
    description="Monthly data pipeline, model training, inference and monitoring for the credit default model",
    default_args=default_args,
    schedule="0 0 1 * *",              # monthly, first of the month
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2025, 7, 1),     # last month with new labels (applications until 2025-01 + 6 mob)
    catchup=True,
    max_active_runs=1,                 # months run in order: training needs the earlier label partitions
    tags=["mle", "credit", "assignment2"],
) as dag:

    start = EmptyOperator(task_id="start")

    # ---------------- data pipeline ----------------
    label_pipeline = BashOperator(
        task_id="label_pipeline",
        bash_command=script('python3 -m utils.airflow_data_pipeline --snapshotdate "{{ ds }}" --stage labels'),
        skip_on_exit_code=99,
    )

    feature_pipeline = BashOperator(
        task_id="feature_pipeline",
        bash_command=script('python3 -m utils.airflow_data_pipeline --snapshotdate "{{ ds }}" --stage features'),
        skip_on_exit_code=99,
    )

    # ---------------- model training ----------------
    retrain_gate = ShortCircuitOperator(
        task_id="retrain_gate",
        python_callable=is_retrain_date,
        ignore_downstream_trigger_rules=False,   # skip only model_train; inference/monitoring still run
    )

    model_train = BashOperator(
        task_id="model_train",
        bash_command=script('python3 -m utils.airflow_model_train --model_train_date "{{ ds }}" '
                            '--train_test_months 12 --oot_months 2'),
    )

    # ---------------- inference ----------------
    model_inference = BashOperator(
        task_id="model_inference",
        bash_command=script('python3 -m utils.airflow_model_inference --snapshotdate "{{ ds }}" --modelname latest'),
        skip_on_exit_code=99,
        trigger_rule=TriggerRule.NONE_FAILED,
    )

    # ---------------- monitoring ----------------
    model_monitoring = BashOperator(
        task_id="model_monitoring",
        bash_command=script('python3 -m utils.airflow_model_monitoring --snapshotdate "{{ ds }}"'),
        skip_on_exit_code=99,
        trigger_rule=TriggerRule.NONE_FAILED,
    )

    end = EmptyOperator(task_id="end", trigger_rule=TriggerRule.NONE_FAILED)

    start >> [label_pipeline, feature_pipeline]
    label_pipeline >> retrain_gate >> model_train
    [feature_pipeline, model_train] >> model_inference
    [model_inference, label_pipeline] >> model_monitoring >> end
