import pandas as pd
import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent))

from src import feature_engineering

def predict(
    sample_users,
    members,
    transactions,
    user_logs_path,
    model,
    feature_columns,
    cutoff_date="2017-03-31",
    id_col="msno",
    target_col="is_churn",
):
    if sample_users[id_col].duplicated().any():
        raise ValueError(f"{id_col} must be unique in sample_users")

    print("Building features...", flush=True)
    features = feature_engineering.build_features(
        sample_users,
        members,
        transactions,
        cutoff_date=cutoff_date,
        path_to_user_logs=user_logs_path,
    )

    features = feature_engineering.features_engineering(
        features,
        cutoff=cutoff_date,
    )

    missing_columns = set(feature_columns) - set(features.columns)
    if missing_columns:
        raise ValueError(f"Missing model features: {sorted(missing_columns)}")

    print("Generating predictions...", flush=True)
    X = features.loc[:, feature_columns]

    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError("The fitted model does not contain churn class 1")

    churn_class_index = classes.index(1)
    probabilities = model.predict_proba(X)[:, churn_class_index]

    predictions = pd.DataFrame({
        id_col: features[id_col].to_numpy(),
        target_col: probabilities,
    })

    result = sample_users[[id_col]].merge(
        predictions,
        on=id_col,
        how="left",
        validate="one_to_one",
        sort=False,
    )

    if result[target_col].isna().any():
        raise ValueError("Some users do not have a prediction")

    if not np.isfinite(result[target_col]).all():
        raise ValueError("Predictions contain non-finite values")

    return result