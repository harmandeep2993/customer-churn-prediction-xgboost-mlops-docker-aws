from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from src.utils.logger import get_logger

class ModelTrainer:
    """Handles training and tuning of XGBoost model."""

    def __init__(self):
        self.logger = get_logger(__name__)

    def train_xgb_model(self, X_train, y_train, n_iter=15, scoring='recall'):
        """
        Train an XGBoost classifier using RandomizedSearchCV optimized for recall.

        Parameters:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training labels.
            n_iter (int): Number of random search iterations.
            scoring (str): Evaluation metric (default = recall).

        Returns:
            tuple: (best_model, best_params)
        """
        self.logger.info("Initializing XGBoost model training with hyperparameter tuning (recall focus).")

        # Define parameter space tuned for recall and imbalance
        param_dist = {
            "n_estimators": [200, 300, 400],
            "max_depth": [3, 4, 5, 6],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "scale_pos_weight": [2, 3, 4],  # important for class imbalance
        }

        model = XGBClassifier(
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=42,
            tree_method="hist",
        )

        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring=scoring,
            cv=3,
            verbose=1,
            random_state=42,
            n_jobs=-1,
        )

        try:
            search.fit(X_train, y_train)
            best_model = search.best_estimator_
            best_params = search.best_params_
            self.logger.info(f"Model training complete. Best parameters: {best_params}")
            return best_model, best_params

        except Exception as e:
            self.logger.error(f"Error during model training: {e}")
            raise

if __name__ == "__main__":
    print("ModelTrainer module ready for import.")