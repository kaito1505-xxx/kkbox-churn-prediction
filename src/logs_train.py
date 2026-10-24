import pandas as pd
import numpy as np
import duckdb
from pathlib import Path
import tempfile

def extract_raw_log_aggregates(
    target_users,
    cutoff_date=None,
    path="../data/interim/user_logs.parquet",
    memory_limit="4GB",
    temp_directory="../data/interim/duckdb_tmp",
):
    cutoff = int(pd.Timestamp(cutoff_date).strftime("%Y%m%d"))

    target_path = Path(tempfile.gettempdir()) / "target_users.parquet"
    pd.DataFrame({
        "msno": pd.Series(target_users, dtype="string").drop_duplicates()
    }).to_parquet(target_path, index=False)

    temp_directory = Path(temp_directory)
    temp_directory.mkdir(parents=True, exist_ok=True)
    
    con = duckdb.connect()
    con.execute(f"SET memory_limit = '{memory_limit}'")
    con.execute(f"SET temp_directory = '{temp_directory}'")

    try:
        con.execute("""
            CREATE TEMP TABLE target_users AS
            SELECT DISTINCT msno
            FROM read_parquet(?)
        """, [str(target_path)])

        result = con.execute("""
            SELECT
                logs.msno,
                SUM(COALESCE(logs.num_25, 0)) AS total_num_25,
                SUM(COALESCE(logs.num_50, 0)) AS total_num_50,
                SUM(COALESCE(logs.num_75, 0)) AS total_num_75,
                SUM(COALESCE(logs.num_985, 0)) AS total_num_985,
                SUM(COALESCE(logs.num_100, 0)) AS total_num_100,
                SUM(COALESCE(logs.num_unq, 0)) AS total_num_unq,
                SUM(
                    LEAST(
                        GREATEST(COALESCE(logs.total_secs, 0), 0),
                        86400
                    )
                ) AS total_secs,
                MAX(logs.date) AS last_activity,
                COUNT(DISTINCT logs.date) AS active_days
            FROM read_parquet(?) AS logs
            INNER JOIN target_users AS targets USING (msno)
            WHERE logs.date <= ?
            GROUP BY logs.msno
        """, [str(path), cutoff]).df()

        result["last_activity"] = pd.to_datetime(
            result["last_activity"].astype("Int64").astype("string"),
            format="%Y%m%d",
            errors="coerce",
        )
        return result

    finally:
        con.close()
        Path(target_path).unlink(missing_ok=True)
        
        
def process_log_features(raw_features):
    df = raw_features.copy()
    
    df["total_plays"] = (
        df["total_num_25"]
        + df["total_num_50"]
        + df["total_num_75"]
        + df["total_num_985"]
        + df["total_num_100"]
    )
    
    safe_active_days = df["active_days"].replace(0, np.nan)
    safe_total_plays = df["total_plays"].replace(0, np.nan)
    safe_unique_tracks = df["total_num_unq"].replace(0, np.nan)

    df["skip_rate"] = (
        df["total_num_25"] / safe_total_plays
    )

    df["completed_rate"] = (
        df["total_num_100"] / safe_total_plays
    )

    df["plays_per_unique_track"] = (
        df["total_plays"] / safe_unique_tracks
    )

    df["plays_per_active_day"] = (
        df["total_plays"] / safe_active_days
    )

    df["unique_tracks_per_active_day"] = (
        df["total_num_unq"] / safe_active_days
    )

    df["secs_per_active_day"] = (
        df["total_secs"] / safe_active_days
    )

    df["secs_per_play"] = (
        df["total_secs"] / safe_total_plays
    )

    return df
