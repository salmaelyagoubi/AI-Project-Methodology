"""
This module contains functions for making predictions and evaluating the model.
"""
from sklearn.metrics import accuracy_score


def predict(model, X_test):
    """
    Make predictions using the given model.

    Args:
        model (sklearn.base.BaseEstimator): The trained model.
        X_test (numpy.ndarray): The test data.

    Returns:
        numpy.ndarray: The predicted labels.
    """
    y_pred = model.predict(X_test)
    return y_pred


def evaluate_model(y_test, y_pred):
    """
    Evaluate the model's accuracy.

    Args:
        y_test (numpy.ndarray): The true labels.
        y_pred (numpy.ndarray): The predicted labels.

    Returns:
        float: The model's accuracy.
    """
    accuracy = accuracy_score(y_test, y_pred) * 100
    return accuracy