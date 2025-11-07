from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

def evaluate_model(model, X_test, y_test):
    """
    Evaluate a classification model and print key metrics.

    Parameters:
    - model: trained classifier
    - X_test: test feature data
    - y_test: true labels

    Returns:
    - metrics: dictionary with scores
    - cm: confusion matrix
    """
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
    }

    cm = confusion_matrix(y_test, y_pred)

    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred))

    print("=== Confusion Matrix ===")
    print(cm)

    return metrics, cm