import pandas as pd

def get_feature_columns(X):
    
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_cols = X.select_dtypes(include=['number', 'bool']).columns.tolist()
    
    return numeric_cols, categorical_cols