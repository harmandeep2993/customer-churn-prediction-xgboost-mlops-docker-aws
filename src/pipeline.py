import pandas as pd
from src.preprocess import (drop_columns, encode_binary_columns, convert_total_charges, one_hot_encode, scale_numerical)

def full_preprocess_pipeline(df):

    # Run full preprocessing on a raw churn dataset.
    df = convert_total_charges(df)
    df = encode_binary_columns(df)

    # Optional: drop customerID if still present
    if "customerID" in df.columns:

        df = drop_columns(df, ["customerID"])

    # One-hot encode remaining categoricals
    cat_cols = df.select_dtypes(include=["object", "bool"]).columns.tolist()
    df = one_hot_encode(df, cat_cols)

    # Scale numeric columns except target
    num_cols = df.select_dtypes(include=["int64", "float64"]) \
                 .drop(columns=["Churn"], errors="ignore") \
                 .columns.tolist()
    df = scale_numerical(df, num_cols)

    return df