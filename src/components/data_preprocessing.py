import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.utils.logger import get_logger


class DataPreprocessing:
    """Handles all preprocessing steps for churn dataset."""

    def __init__(self):
        self.logger = get_logger(__name__)

    def convert_total_charges(self, df):
        """Convert TotalCharges to numeric and fill missing with 0."""
        self.logger.info("Converting TotalCharges to numeric and filling missing values.")
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
        return df

    def drop_columns(self, df, columns):
        """Drop specified columns safely."""
        self.logger.info(f"Dropping columns: {columns}")
        return df.drop(columns=columns, errors='ignore')

    def encode_binary_columns(self, df):
        """Encode Yes/No binary columns as 1/0."""
        self.logger.info("Encoding binary Yes/No columns.")
        binary_cols = [
            col for col in df.columns
            if df[col].nunique() == 2 and df[col].dropna().isin(['Yes', 'No']).all()
        ]
        df[binary_cols] = df[binary_cols].replace({'Yes': 1, 'No': 0})
        return df

    def scale_numeric_columns(self, df, cols_to_scale):
        """Scale numeric columns using StandardScaler."""
        self.logger.info(f"Scaling numeric columns: {cols_to_scale}")
        scaler = StandardScaler()
        for col in cols_to_scale:
            if col in df.columns:
                df[[col]] = scaler.fit_transform(df[[col]])
        return df

    def full_preprocess_pipeline(self, df, encoder=None, fit_encoder=False):
        """
        Apply preprocessing steps consistent with training.
        """
        df = df.copy()
        self.logger.info("Starting full preprocessing pipeline.")

        # Convert TotalCharges
        df = self.convert_total_charges(df)

        # Drop customerID
        if 'customerID' in df.columns:
            df = self.drop_columns(df, ['customerID'])

        # Encode gender
        if 'gender' in df.columns:
            df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})

        # Encode binary columns
        df = self.encode_binary_columns(df)

        # Identify multi-category categorical columns
        cat_cols = [
            c for c in df.select_dtypes(include=['object']).columns
            if df[c].nunique() > 2
        ]
        self.logger.info(f"Categorical columns to encode: {cat_cols}")

        # Fit or use existing encoder
        if fit_encoder:
            self.logger.info("Fitting OneHotEncoder.")
            encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            encoded = encoder.fit_transform(df[cat_cols])
        else:
            if encoder is None:
                self.logger.error("Encoder must be provided when fit_encoder=False")
                raise ValueError('Encoder must be provided when fit_encoder=False')
            encoded = encoder.transform(df[cat_cols])

        encoded_df = pd.DataFrame(
            encoded,
            columns=encoder.get_feature_names_out(cat_cols),
            index=df.index,
        )

        df = pd.concat([df.drop(columns=cat_cols), encoded_df], axis=1)

        # Scale numeric columns
        cols_to_scale = ['tenure', 'MonthlyCharges', 'TotalCharges']
        df = self.scale_numeric_columns(df, cols_to_scale)

        # Convert object columns to numeric codes
        for c in df.columns:
            if df[c].dtype == 'object':
                df[c] = df[c].astype('category').cat.codes

        self.logger.info("Preprocessing completed successfully.")
        if fit_encoder:
            return df, encoder
        return df


if __name__ == '__main__':
    import joblib
    from src.components.data_ingestion import DataIngestion

    ingestion = DataIngestion()
    df = ingestion.load_dataset()

    preprocessor = DataPreprocessing()
    processed_df, encoder = preprocessor.full_preprocess_pipeline(df, fit_encoder=True)

    joblib.dump(encoder, 'models/onehot_encoder.pkl')
    processed_df.to_csv('data/processed/churn_cleaned.csv', index=False)
    print("Preprocessing complete.")