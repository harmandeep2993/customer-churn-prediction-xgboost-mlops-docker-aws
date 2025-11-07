import pandas as pd
from sklearn.preprocessing import StandardScaler

# Function to columns 
def drop_columns(df, columns):
    
    #Drop specified columns from DataFrame.
    return df.drop(columns=columns, axis=1)

# Function to encode binary features
def encode_binary_columns(df):
    
    # Separate the binary features to map them
    binary_cols = [col for col in df.columns if df[col].nunique() == 2 and df[col].dtype == 'object']
    
    for col in binary_cols:

        # Map 'Yes'/'No' binary columns to 1/0.
        df[col] = df[col].map({'Yes': 1, 'No': 0})

    return df

# Function to convert total_charges into int type
def convert_total_charges(df):
    
    #Convert TotalCharges to numeric, coerce errors to NaN.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df

# Fucntion to convert categorical features
def one_hot_encode(df, columns):
    
    # Apply one-hot encoding to specified categorical columns.
    return pd.get_dummies(df, columns=columns, drop_first=True)

# Fucntion to scale down values
def scale_numerical(df, columns):
    
    # Standardize numerical columns using StandardScaler.
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df
