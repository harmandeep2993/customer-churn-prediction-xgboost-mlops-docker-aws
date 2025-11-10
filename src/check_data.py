import pandas as pd

RAW_DATA_PATH = './data/raw/Customer-Churn.csv'


def load_dataset(path=RAW_DATA_PATH):
    """Load dataset from CSV file."""
    return pd.read_csv(path)


def df_overview(df):
    """Print dataset overview: shape, data types, and missing values."""
    print('\n=== Shape ===')
    print(df.shape)

    print('\n=== Dtypes ===')
    print(df.dtypes)

    print('\n=== Missing Values ===')
    print(df.isna().sum().sort_values(ascending=False).head(20))


if __name__ == '__main__':
    df = load_dataset()
    df_overview(df)