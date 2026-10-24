import pandas as pd
import numpy as np

def filter_transactions_by_cutoff(transactions, cutoff_date="2017-01-31"):
    
    df_trans = transactions.copy()
    
    df_trans["transaction_date"] = pd.to_datetime(
        df_trans["transaction_date"].astype("string"), 
        format="%Y%m%d", 
        errors="coerce"
    )
    
    cutoff = pd.to_datetime(cutoff_date, errors="raise")
    
    df_trans = df_trans.loc[df_trans["transaction_date"] <= cutoff]
    
    return df_trans

def aggregate_transactions(df):
    
    df['discount_amount'] = df['plan_list_price'] - df['actual_amount_paid']
    
    df['is_discount'] = (df['discount_amount'] > 0).astype(int)

    trans_features = df.groupby('msno').agg({
        
        'payment_method_id': lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0],
        
        'payment_plan_days': 'mean',
        'plan_list_price': 'median',
        'actual_amount_paid': 'median',

        'discount_amount': 'mean', 
        'is_discount': 'mean', 
        
        'is_auto_renew': 'mean',
        'transaction_date': ['min', 'max', 'count'],
        'is_cancel': 'mean' 
        
    })

    trans_features.columns = [
        'payment_method_most_frequent',
        'plan_days_mean',
        'plan_list_price_median',
        'actual_amount_paid_median',
        'discount_amount_mean',
        'is_discount_rate',
        'auto_renew_rate',
        'first_transaction',
        'last_transaction',
        'total_transactions',
        'is_cancel_rate'
    ]

    trans_features['transaction_history_days'] = (trans_features['last_transaction'] - trans_features['first_transaction']).dt.days
    
    return trans_features.reset_index()