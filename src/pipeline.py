import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from src.preprocess import full_preprocess_pipeline
from src.train_model import train_xgb_model
from src.evaluate_model import evaluate_model

# 1. Load raw data
raw_data_path = './data/raw/Customer-Churn.csv'
df = pd.read_csv(raw_data_path)

# 2. Preprocess data
df, encoder = full_preprocess_pipeline(df, fit_encoder=True)
df.to_csv('./data/processed/churn_cleaned.csv', index=False)
print('Data preprocessed and saved to ./data/processed/churn_cleaned.csv')

# Save encoder for reuse
joblib.dump(encoder, 'models/onehot_encoder.pkl')
print('Encoder saved to models/onehot_encoder.pkl')

# 3. Split data
X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print('Data split into training and test sets')

# 4. Define hyperparameter space
param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'scale_pos_weight': [1, 2]
}

# 5. Train and tune model
model, best_params = train_xgb_model(X_train, y_train, param_dist)
print('Model training complete')
print(f'Best Parameters: {best_params}')

# 6. Save model and training columns
joblib.dump(model, 'models/xgb_churn_full_tuned.pkl')
joblib.dump(X_train.columns.tolist(), 'models/train_columns.pkl')
print('Model and feature list saved to models/')

# 7. Evaluate model
print('=== Model Evaluation ===')
evaluate_model(model, X_test, y_test)