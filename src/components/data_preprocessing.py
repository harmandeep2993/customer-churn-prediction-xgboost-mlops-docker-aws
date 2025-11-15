import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.utils.logger import get_logger


class DataPreprocessing:
    """Handles preprocessing for churn dataset."""

    def __init__(self):
        self.logger = get_logger(__name__)
        self.cat_cols = ["MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
            "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract", "PaymentMethod"
]
        self.num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

    def convert_total_charges(self, df):
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
        return df

    def full_preprocess_pipeline(self, df, encoder=None, fit_encoder=False):
        df = df.copy()
        self.logger.info("Starting full preprocessing pipeline.")

        # Convert TotalCharges
        df = self.convert_total_charges(df)

        # Drop ID if present
        if "customerID" in df.columns:
            df = df.drop(columns=["customerID"])

        # Encode gender manually
        if "gender" in df.columns:
            df["gender"] = df["gender"].map({"Male": 1, "Female": 0})

        # Encode binary Yes/No columns
        binary_cols = [ c for c in ["SeniorCitizen", "Partner", "Dependents", "PhoneService", "PaperlessBilling"] 
            if c in df.columns]
        
        df[binary_cols] = df[binary_cols].replace({"Yes": 1, "No": 0})

        # OneHotEncode fixed categorical columns
        cat_cols = [c for c in self.cat_cols if c in df.columns]
        if fit_encoder:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            encoded = encoder.fit_transform(df[cat_cols])
        else:
            encoded = encoder.transform(df[cat_cols])

        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols), index=df.index)

        # Merge numeric + binary + encoded
        df = pd.concat([df.drop(columns=cat_cols), encoded_df], axis=1)

        # Scale numeric features
        scaler = StandardScaler()
        df[self.num_cols] = scaler.fit_transform(df[self.num_cols])

        self.logger.info("Preprocessing completed successfully.")
        return (df, encoder) if fit_encoder else df