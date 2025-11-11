# Customer Churn Prediction
An end-to-end machine learning project that predicts customer churn using XGBoost. Built with a modular pipeline, YAML configuration, and a Streamlit web app for deployment. This project demonstrates data preprocessing, model training, evaluation, and deployment using clean and maintainable code.

## Objective
Predict whether a telecom customer will churn (cancel service) based on demographic, billing, and usage data. The goal is to help businesses retain customers and improve customer lifetime value.

## Dataset Source
Telco Customer Churn Dataset from Kaggle  
[https://www.kaggle.com/blastchar/telco-customer-churn](https://www.kaggle.com/blastchar/telco-customer-churn)

## Business Impact
- Retaining existing customers is cheaper than acquiring new ones.  
- Early churn detection helps in offering targeted discounts or support.  
- Predictive analytics improves decision-making and profitability.

## Project Structure

```bash
customer-churn-prediction/
│
├── app/
│   └── streamlit_app.py              # Streamlit app for predictions
│
├── config/
│   ├── config.yaml                   # Configuration file for paths and parameters
│   └── __init__.py
│
├── data/
│   ├── raw/                          # Original dataset
│   └── processed/                    # Processed dataset
│
├── models/
│   ├── xgb_churn_full_tuned.pkl      # Trained model
│   ├── onehot_encoder.pkl            # Saved encoder
│   └── train_columns.pkl             # Training column list
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_model_training.ipynb
│
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_preprocessing.py
│   │   ├── model_trainer.py
│   │   ├── model_evaluator.py
│   │   ├── model_predictor.py
│   │   └── pipeline.py
│   │
│   ├── utils/
│   │   ├── logger.py
│   │   ├── config_loader.py
│   │   └── file_handler.py
│   │
│   └── __init__.py
│
├── setup.py
├── requirements.txt
├── .gitignore
└── README.md
````

## Workflow
   1. Data Ingestion
      Load and validate raw customer churn data from CSV.

   2. Data Preprocessing
      Encode categorical features, scale numeric features, and handle missing values.
      Save the processed data and encoder for reuse.

   3. Model Training
      Train an XGBoost classifier using RandomizedSearchCV for hyperparameter tuning.
      Handle class imbalance using the `scale_pos_weight` parameter.

   4. Model Evaluation
      Evaluate performance using Accuracy, Precision, Recall, F1-score, and Confusion Matrix.
      Select the best model based on F1-score.

   5. Deployment
      Use Streamlit for real-time churn prediction based on user input.

## Model Performance

   | Metric            | Round 1 | Round 2 (Final) |
   | ----------------- | ------- | --------------- |
   | Accuracy          | 0.75    | 0.78            |
   | Recall (Churn)    | 0.81    | 0.73            |
   | Precision (Churn) | 0.52    | 0.56            |
   | F1 (Churn)        | 0.64    | 0.63            |

Final Model: Tuned XGBoost (Round 2) with better balance between precision and recall.

## Key Learnings

   * Built a modular and reusable ML pipeline.
   * Handled class imbalance effectively.
   * Implemented logging and configuration management.
   * Deployed the final model with Streamlit.
   * Practiced clean, structured, and reproducible code.

## Tech Stack

   * Python 3.10+
   * Pandas, NumPy, Scikit-learn, XGBoost
   * Matplotlib, Seaborn
   * Streamlit, Joblib, PyYAML, Logging

## How to Run Locally

1. Clone the Repository

```bash
git clone https://github.com/<your-username>/customer-churn-prediction.git
cd customer-churn-prediction
```

2. Create and Activate Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate     # On Windows: .venv\Scripts\activate
```

3. Install Dependencies

```bash
pip install -r requirements.txt
```

4. Run the Training Pipeline

```bash
python -m src.components.pipeline
```

5. Launch the Streamlit App

```bash
streamlit run app/streamlit_app.py
```

## Example Output

```
Prediction Result:
No Churn
Churn Probability: 22.50%
```

## Future Improvements

   * Add explainability using SHAP or LIME.
   * Automate retraining with live data.
   * Store artifacts in cloud storage (AWS S3).
   * Containerize using Docker for production deployment.

## Author
Harmandeep Singh
Machine Learning and Data Science Enthusiast
Based in Germany

- [LinkedIn](https://www.linkedin.com/in/harmandeep/) 
- [GitHub](https://github.com/harmandeep2993)