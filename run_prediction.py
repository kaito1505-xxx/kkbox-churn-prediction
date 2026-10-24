import joblib
import pandas as pd
from pathlib import Path

from src.prediction import predict


ROOT = Path(__file__).resolve().parent

DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"
MODEL_PATH = ROOT / "models" / "lightgbm_pipeline.joblib"

print("Loading sample users...", flush=True)
sample_users = pd.read_csv(
    DATA_RAW / "sample_submission_v2.csv",
    usecols=["msno"],
)

print("Loading members...", flush=True)
members = pd.read_csv(DATA_RAW / "members_v3.csv")

print("Loading transactions...", flush=True)
transactions = pd.read_csv(DATA_INTERIM / "transactions.csv")

model_artifact = joblib.load(MODEL_PATH)

predictions = predict(
    sample_users=sample_users,
    members=members,
    transactions=transactions,
    user_logs_path=DATA_INTERIM / "user_logs.parquet",
    model=model_artifact["model"],
    feature_columns=model_artifact["feature_columns"],
    cutoff_date="2017-03-31",
    id_col=model_artifact.get("id_column", "msno"),
    target_col=model_artifact.get("target_column", "is_churn"),
)

print("Saving predictions...", flush=True)

output_dir = DATA_PROCESSED / "predictions"
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "april_2017_predictions.csv"
predictions.to_csv(output_path, index=False)

print(f"Saved {len(predictions):,} predictions to {output_path}")