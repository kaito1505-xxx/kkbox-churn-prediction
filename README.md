# KKBox Subscription Churn Prediction

A machine learning project that predicts whether a KKBox user will churn when their subscription expires. It covers time-aware feature engineering, model training, temporal validation, and batch probability generation for the unlabeled April 2017 cohort.

## Project goal

For each user whose subscription is approaching expiration, estimate the probability that they will **not renew within 30 days after expiration**.

The target is `is_churn`:

- `1` — the user did not continue the subscription within 30 days of expiration.
- `0` — the user renewed within 30 days.

The April 2017 users in `sample_submission_v2.csv` have no target labels. This project generates churn probabilities for them, but their actual performance cannot be measured without ground-truth labels.

## Why the feature cutoff matters

A churn model should use only information that would have been available when the prediction is made. Including transactions or listening activity recorded after the prediction date would introduce temporal data leakage.

Features are built using data available up to the end of the month before the target cohort:

| Cohort | Subscription expiry month | Feature cutoff |
|---|---|---|
| Train 1 | February 2017 | January 31, 2017 |
| Train 2 | March 2017 | February 28, 2017 |
| April prediction cohort | April 2017 | March 31, 2017 |

The cutoff applies to dated transaction and user-log events. It does not change the target: `is_churn` still describes what happened during the 30 days after subscription expiration.

## Data

The project uses the KKBox churn challenge data, including:

- user membership information;
- transaction history;
- user listening logs;
- labeled February and March cohorts;
- `sample_submission_v2.csv`, which provides the April cohort user IDs and the expected output format.

The `is_churn` values in the sample submission file are placeholders, not labels. They are not used as training targets.

Raw and processed challenge data are not included in this repository. Obtain the data from its original source and place the files in the expected local directories. Do not commit challenge data or model artifacts unless their license and size permit redistribution.

The prediction container expects these local directories to be available:

```text
data/
├── raw/
├── interim/
└── processed/
    └── predictions/
models/
```

Place the required source data in `data/raw/` and `data/interim/`, and the trained model artifact in `models/`. The exact filenames must match the paths configured in the prediction code.

## Modeling and evaluation

The modeling notebook compares several classifiers and tunes LightGBM using Average Precision (AP). AP is used because the target is imbalanced and the project is interested in ranking users by churn risk.

The best reported result was:

- **Model:** LightGBM after `RandomizedSearchCV`
- **Average Precision on Train 2:** approximately `0.526`
- **Positive-class prevalence in Train 2:** approximately `0.090`

Train 2 is a later cohort than Train 1, so it provides a useful temporal validation scenario. However, Train 2 was inspected during model comparisons and threshold exploration. Its score should therefore be interpreted as a **temporal validation result**, not as an untouched, independent test estimate.

The April cohort is unlabeled, so its predictions cannot be evaluated in this project.

## Predictions

The prediction pipeline produces a probability for each user in the April cohort:

- values closer to `1` indicate higher estimated churn risk;
- values closer to `0` indicate lower estimated churn risk.

The output keeps probabilities rather than converting them to binary `0`/`1` decisions. A binary decision would require a threshold chosen for a specific operational goal, such as a target recall or the cost of contacting users.

The generated CSV has these columns:

```text
msno,is_churn
```

Here, `is_churn` contains predicted probabilities, not confirmed outcomes or binary labels.

## Docker batch prediction

Docker packages the prediction code and its Python dependencies so the batch prediction can run in a consistent environment. The data and model are mounted into the container at runtime; they do not need to be copied into the Docker image.

### Build the image

From the project root—the directory containing the `Dockerfile`—run:

```bash
docker build -t kkbox-churn:local .
```

### Prepare the output directory

```bash
mkdir -p data/processed/predictions
```

### Run the prediction

```bash
docker run --rm --name kkbox-predict \
  --mount type=bind,source="$(pwd)/data/raw",target=/app/data/raw,readonly \
  --mount type=bind,source="$(pwd)/data/interim",target=/app/data/interim,readonly \
  --mount type=bind,source="$(pwd)/models",target=/app/models,readonly \
  --mount type=bind,source="$(pwd)/data/processed/predictions",target=/app/data/processed/predictions \
  kkbox-churn:local \
  python -u run_prediction.py
```

The input directories and model directory are mounted read-only. The predictions directory is writable so the container can save the output CSV to the host machine.

The command uses `--rm`, so Docker removes the container after it exits. The generated prediction file remains in `data/processed/predictions/` on the host.

The `-u` option runs Python with unbuffered output, which makes progress messages appear promptly in the terminal. If the container stops unexpectedly, inspect its logs and check whether Docker reports `OOMKilled=true`. Feature aggregation of the large user-log dataset can use substantial memory; the Docker memory limit and the DuckDB temporary-file directory must be configured for the available system resources and disk space.

### Rebuild after code or dependency changes

If you change the Dockerfile, dependency file, or code copied into the image, rebuild it:

```bash
docker build -t kkbox-churn:local .
```

Changes to files mounted from the host, such as input data or model artifacts, are available to the container without rebuilding the image.

## Repository workflow

The notebooks are intended to be run in order:

1. Data exploration and quality checks.
2. Feature engineering for the labeled cohorts.
3. Feature and target preparation.
4. Model comparison, parameter search, and temporal validation.
5. April cohort prediction.

The prediction pipeline loads the previously trained model artifact, builds April features using the March 31 cutoff, and generates probabilities for the user IDs in `sample_submission_v2.csv`.

Docker is used for this batch prediction workflow.
## Reproducibility notes

- Use the same feature columns and preprocessing pipeline during training and prediction.
- Keep the cohort-specific cutoffs when constructing training features.
- Do not use April sample-submission placeholder values as labels.
- Keep raw data and model artifacts outside the Docker image; mount them at runtime.
- Use compatible Python and package versions when loading a serialized model.
- Exact results may vary with library versions, hardware, and training configuration.

## Limitations

- Train 2 was used during model development, so its result is not a fully independent estimate.
- The April cohort has no labels available in this project; April prediction quality is unknown.
- Predicted probabilities are risk scores. Unless calibration is evaluated separately, they should not automatically be interpreted as perfectly calibrated real-world probabilities.