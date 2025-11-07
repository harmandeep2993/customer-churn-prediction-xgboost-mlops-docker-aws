import pandas as pd
from src.preprocess import (drop_columns, encode_binary_columns, convert_total_charges, one_hot_encode, scale_numerical)

# Define function for full transformation
def full_preprocess_pipeline(df):
    '''
    Apply all preprocessing steps to a raw customer churn DataFrame.
    Returns the cleaned, encoded, and scaled DataFrame.

    '''

    # Convert 'TotalCharges' to numeric, coerce invalid values to NaN
    df = convert_total_charges(df)

    # Encode binary categorical columns ('Yes'/'No') to 1/0
    df = encode_binary_columns(df)

    # Drop identifier column if present (e.g., 'customerID')
    if "customerID" in df.columns:
        df = drop_columns(df, ["customerID"])

    # Identify categorical columns with >2 unique values
    cat_cols = df.select_dtypes(include=["object", "bool"]).columns.tolist()

    # Apply one-hot encoding to multi-class categorical features
    df = one_hot_encode(df, cat_cols)

    # Identify numerical columns (excluding target 'Churn') for scaling
    num_cols = df.select_dtypes(include=["int64", "float64"]).drop(columns=["Churn"], errors="ignore").columns.tolist()

    # Apply standard scaling to numerical features
    df = scale_numerical(df, num_cols)

    return df