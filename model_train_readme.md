# How Training, Inference and Monitoring Work

This guide explains the modelling stages of the credit default pipeline as the code works today. For setup and running the project, see `README.md`.

| Stage | Script | Logic | Runs | Output |
|---|---|---|---|---|
| Training | `model_train.py` | `utils/model_train.py` | Dates in `RETRAIN_DATES` | `model_bank/` |
| Inference | `model_inference.py` | Same file | Every month | `datamart/gold/model_predictions/` |
| Monitoring | `model_monitoring.py` | `utils/model_monitoring.py` | Every month | `datamart/gold/model_monitoring/` |

## Key dates

A loan has two dates, and all three stages depend on the difference:

- **Application date:** `loan_start_date`, which is when the model makes its prediction. Features are taken from this month.
- **Label date:** `snapshot_date` in the label store, which is when the outcome becomes known, 6 months after the loan starts.

For example, a loan that starts in 2024-01 is scored in 2024-01, but its label isn't known until 2024-07.

The label is `30dpd_6mob`: 1 if the loan is 30 or more days past due at 6 months on book, otherwise 0.

## Training

### Why the model trains on 2024-09-01

`RETRAIN_DATES` in `dag/dag.py` contains only `2024-09-01`, the first date with enough mature labels for the default 12 + 2 month window:

1. The first loans start in 2023-01, so the first labels are known in 2023-07.
2. The window needs 14 months of labels, which runs from 2023-07 to 2024-08.
3. Training uses only labels known before its date, so the earliest date is 2024-09-01.

A shorter window would allow an earlier date. For example, 6 + 2 months allows 2024-03-01.

### Data windows

Windows are cut on the label date and counted back from the training date:

| Split | Label months | Application months | Rows | Purpose |
|---|---|---|---|---|
| Train | 2023-07 to 2024-06 | 2023-01 to 2023-12 | 4,766 | Tuning and fitting |
| Test | 2023-07 to 2024-06 | 2023-01 to 2023-12 | 1,192 | Choosing between models |
| Out-of-time (OOT) | 2024-07 to 2024-08 | 2024-01 to 2024-02 | 1,003 | Final check and monitoring baseline |

Train and test are a random, stratified 80/20 split of the same 12 months. OOT comes later in time, so it's the better guide to future performance.

### Steps

1. **Load labels:** read labels whose label date falls inside the window.
2. **Join features:** left join the feature store on `Customer_ID` and application date.
3. **Run safety checks:** stop with an error if the window has no labels, any label has no features, or any feature is dated on or after the training date.
4. **Choose inputs:** remove ID, date and label columns, plus `Payment_Behaviour` and `Credit_Mix_MinAmt`, which are redundant. This leaves 57 features.
5. **Split the data:** create the train, test and OOT sets above.
6. **Tune candidates:** run `RandomizedSearchCV` on the train set for each candidate, with up to 20 draws, 3-fold CV and ROC AUC scoring.
7. **Select the winner:** keep the candidate with the highest test AUC. OOT is reported but not used to choose.
8. **Build the monitoring baseline:** score the test set and store the distributions of the scores and of each feature.
9. **Save the model:** write the `.pkl` and `.json` files to `model_bank/`.

### Preprocessing

Preprocessing is fitted on the training folds and saved inside the model pipeline, so inference applies exactly the same steps.

| Column type | Logistic regression | XGBoost |
|---|---|---|
| Numeric | Median fill, missing flag, scaling | Median fill and missing flag, no scaling |
| Clickstream (`fe_*`) | Treated like other numeric columns | Left as missing, handled natively by XGBoost |
| Categorical | `_` and nulls become `Unknown`, then one-hot encoding | Same |

Categories not seen in training are ignored at inference.

### Results for `credit_model_2024_09_01`

XGBoost was selected:

| Model | Train Gini | Test Gini | OOT Gini |
|---|---|---|---|
| **XGBoost** | 0.834 | 0.688 | 0.653 |
| Logistic regression | 0.599 | 0.621 | 0.564 |

XGBoost's gap between train and test suggests some overfitting. The model selection doesn't currently check for this.

### Saved files

For each model, `model_bank/` holds:

- `credit_model_<date>.pkl`: the fitted pipeline, feature lists, date windows, data statistics, metrics, best parameters, candidate results and monitoring baseline.
- `credit_model_<date>.json`: the same information without the fitted model, so it's easy to read.

To run training by hand:

```bash
python model_train.py --model_train_date 2024-09-01 --train_test_months 12 --oot_months 2
```

## Inference

Inference scores one application month at a time.

### Steps

1. **Pick a model:** with `--modelname latest`, take the newest file in `model_bank/` trained on or before the snapshot date. A model can't score a month earlier than its training date this way.
2. **Check there's work to do:** exit with code 99 if there's no model or feature partition, and Airflow marks the task as skipped.
3. **Load features:** read that month's gold feature store partition and convert decimals to doubles, as in training.
4. **Check columns:** stop with an error if any column the model needs is missing.
5. **Score:** run the saved pipeline and keep the probability of default.
6. **Save:** write `Customer_ID`, `snapshot_date`, `model_name` and `model_predictions`, replacing any earlier output for that model and month.

The output is a probability score, not an approve-or-decline decision. The pipeline doesn't apply a cut-off.

### What happens in the DAG

| Run months | Inference result |
|---|---|
| 2023-01 to 2024-08 | Skipped, no model yet |
| 2024-09 to 2025-01 | Scored by `credit_model_2024_09_01` |
| 2025-02 to 2025-07 | Skipped, no new applications |

To score a month by hand:

```bash
python model_inference.py --snapshotdate 2024-10-01 --modelname latest
python model_inference.py --snapshotdate 2024-10-01 --modelname credit_model_2024_09_01.pkl
```

## Monitoring

Monitoring runs every month, including months with no new applications, because labels for earlier applications keep arriving.

### Steps

For each model with predictions, and with a saved `.pkl` in `model_bank/`:

1. **Load predictions:** keep months up to the as-of date.
2. **Add features:** join the features used for those predictions.
3. **Add labels:** join only labels known by the as-of date, which prevents using future outcomes.
4. **Calculate checks:** create one row per application month.
5. **Save outputs:** write a parquet table and a PNG chart for that model and as-of date.

Every run recalculates all scored months, so the latest as-of file gives the complete view.

### What it measures

| Group | Metrics | Available |
|---|---|---|
| Stability | `score_psi`, `max_feature_psi`, `max_psi_feature`, `avg_score` | When the month is scored |
| Data quality | `clickstream_missing` | When the month is scored |
| Performance | `bad_rate`, `auc`, `gini` | 6 months later, with at least 50 labelled rows and both label classes |

`period` shows where each month sits relative to the model's training data:

- `train_test` or `oot`: the model saw these labels during development, so results may be optimistic.
- `post_training`: the labels came after training, so this is the fairest test.

### PSI

Population Stability Index (PSI) measures how far a distribution has moved from its baseline:

```text
PSI = sum of (actual % - expected %) × ln(actual % / expected %)
```

- **Baseline:** the test set, stored in the model file when it's trained.
- **Numeric columns and scores:** 10 quantile bins, plus a separate bin for missing values.
- **Categorical columns:** one bin per category, plus one for unseen values.
- **Empty bins:** proportions have a floor of 0.0001, which avoids dividing by zero.

The floor explains the `fe_*` PSI of 16.11. These columns had no missing values in the baseline, but are 100% missing in the scored months, so the missing bin contributes about 9.2 and the 10 emptied bins contribute about 6.9. Read that value as "this feature is completely missing", not as a measure of gradual drift.

### Status rules

| Check | Green | Amber | Red |
|---|---|---|---|
| Score PSI | ≤ 0.10 | ≤ 0.25 | > 0.25 |
| Highest feature PSI | ≤ 0.10 | ≤ 0.25 | > 0.25 |
| Relative Gini drop against OOT Gini | ≤ 10% | ≤ 20% | > 20% |

`status` is the worst of the three. A check with no result yet shows `N/A` and doesn't affect it.

### Example results

`credit_model_2024_09_01` as of 2025-07-01, against an OOT Gini of 0.653:

| Application month | Clickstream missing | Score PSI | Gini | Relative Gini drop | Gini status | Overall status |
|---|---|---|---|---|---|---|
| 2024-09 | 100% | 0.949 | 0.602 | 8% | GREEN | RED |
| 2024-10 | 100% | 1.051 | 0.622 | 5% | GREEN | RED |
| 2024-11 | 100% | 1.063 | 0.603 | 8% | GREEN | RED |
| 2024-12 | 100% | 0.685 | 0.700 | None (Gini rose) | GREEN | RED |
| 2025-01 | 100% | 1.025 | 0.565 | 13% | AMBER | RED |

All five months are RED because of PSI, even though the Gini check alone is green or amber. The model still ranks customers close to its baseline, but its input data has changed.

Earlier as-of dates have no Gini until labels arrive. For example, the 2024-09-01 table shows 0 labelled rows, an empty Gini and a RED status from PSI.

To run monitoring by hand:

```bash
python model_monitoring.py --snapshotdate 2025-07-01
```

## Current limitations

- **Missing clickstream:** applicants from 2024-07 onward aren't in the clickstream file, so every month this model scores has no clickstream. The model trained only on applicants who had it.
- **No promotion check:** `latest` uses the newest model without comparing it with the previous one.
- **Silent skip:** if training fails, later inference runs skip with "no model" instead of failing.
- **No automated response:** a RED status doesn't send an alert or start retraining.
- **Calibration:** the mean score is about 0.60, while the actual bad rate is about 0.28, so treat scores as a ranking unless they're calibrated first.
