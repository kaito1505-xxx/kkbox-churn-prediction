import pandas as pd
import numpy as np
import src.transactions_train
import src.logs_train

def build_features(users, members, transactions, cutoff_date=None, path_to_user_logs = None):

    target_users = users["msno"]
    
    print("Aggregate user_logs...")
    
    logs_agg = src.logs_train.extract_raw_log_aggregates(
        target_users,
        cutoff_date=cutoff_date, 
        path=path_to_user_logs
    )
    logs_features = src.logs_train.process_log_features(logs_agg)
    
    print("Aggregate transactions...")
    
    transaction_filtered = src.transactions_train.filter_transactions_by_cutoff(transactions, cutoff_date=cutoff_date)
    transaction_features = src.transactions_train.aggregate_transactions(transaction_filtered)

    members_for_merge = members.assign(_member_present=1)
    
    features = (
        users
        .merge(members_for_merge, on="msno", how="left", validate="one_to_one")
        .merge(transaction_features, on="msno", how="left", validate="one_to_one")
        .merge(logs_features, on="msno", how="left", validate="one_to_one")
    )
    
    features["member_missing"] = features["_member_present"].isna().astype("int8")
    features = features.drop(columns="_member_present")
    
    return features

def clean_categories(df, columns):
    df = df.copy()
    for col in columns:
        df[col] = df[col].astype("string").fillna("unknown")
    return df

def features_engineering(df, cutoff):
    df = df.copy()
    
    categorical_cols = ["city", "gender", "registered_via", "payment_method_most_frequent"]
    
    transaction_cols = [
        'plan_days_mean',                     
        'plan_list_price_median',             
        'actual_amount_paid_median',          
        'discount_amount_mean',             
        'is_discount_rate',                 
        'auto_renew_rate',                    
        'total_transactions',                  
        'is_cancel_rate',                      
        'transaction_history_days'
    ]
    
    log_cols = [
        "total_secs",
        "total_num_25",
        "total_num_50",
        "total_num_75",
        "total_num_985",
        "total_num_100",
        "total_num_unq",
        "active_days",
        "total_plays",
        "skip_rate",
        "completed_rate",
        "plays_per_unique_track",
        "plays_per_active_day",
        "unique_tracks_per_active_day",
        "secs_per_active_day",
        "secs_per_play"
    ]
    df["member_missing"] = (
        df["city"].isna()
    ).astype("int8")
    
    df = clean_categories(df, categorical_cols)
    
    df["bd_invalid"] = (
        (df["bd"] <= 0) |
        (df["bd"] > 100)
    ).astype("int8")

    df["bd"] = df["bd"].mask(
        (df["bd"] <= 0) |
        (df["bd"] > 100)
    )
    
    df["registration_init_time"] = pd.to_datetime(
        df["registration_init_time"]
        .astype("Int64")
        .astype("string"),
        format="%Y%m%d",
        errors="coerce"
    )

    df["registration_year"] = (
        df["registration_init_time"].dt.year
    )
    
    cutoff = pd.Timestamp(cutoff)
    
    df["days_since_last_transaction"] = (
        cutoff - df["last_transaction"]
    ).dt.days
    
    df["days_since_last_activity"] = (
        cutoff - df["last_activity"]
    ).dt.days
    
    df["days_since_last_transaction"] = (
        df["days_since_last_transaction"].fillna(-1)
    )
    df["days_since_last_activity"] = (
        df["days_since_last_activity"].fillna(-1)
    )
        
    df[transaction_cols] = (
        df[transaction_cols].fillna(0)
    )
    
    df[log_cols] = df[log_cols].fillna(0)
    
    df["has_user_logs"] = (
        df["active_days"] > 0
    ).astype("int8")
    
    df["has_transactions"] = (
        df["total_transactions"] > 0
    ).astype("int8")
    
    df = df.drop(columns=["registration_init_time", "last_transaction", "first_transaction", "last_activity"])
    
    return df
        

