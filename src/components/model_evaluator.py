from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from src.utils.logger import get_logger


class ModelEvaluator:
    """Handles evaluation of trained classification models."""

    def __init__(self):
        self.logger = get_logger(__name__)

    def evaluate_model(self, model, X_test, y_test):
        """
        Evaluate classification model and print metrics.

        Parameters:
            model: Trained model object.
            X_test (pd.DataFrame): Test features.
            y_test (pd.Series): True labels.

        Returns:
            tuple: (metrics dict, confusion matrix)
        """
        try:
            self.logger.info("Evaluating model performance.")
            y_pred = model.predict(X_test)

            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1_score": f1_score(y_test, y_pred)
            }

            cm = confusion_matrix(y_test, y_pred)

            self.logger.info("Model evaluation complete.")
            self.logger.info(f"Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}")

            print("\n=== Classification Report ===")
            print(classification_report(y_test, y_pred))
            print("=== Confusion Matrix ===")
            print(cm)

            return metrics, cm

        except Exception as e:
            self.logger.error(f"Error during model evaluation: {e}")
            raise

if __name__ == "__main__":
    print("ModelEvaluator module is ready for import.")