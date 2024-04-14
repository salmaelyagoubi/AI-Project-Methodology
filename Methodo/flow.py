"""
This module contains the main workflow for the churn prediction project.
"""
import mlflow
from churn_prediction.preprocess import load_data, preprocess_data
from churn_prediction.train import build_model, train_model, parameter_tuning
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def load_and_preprocess_data(filepath, sheet_name):
    """
    Load and preprocess the data for the churn prediction model.

    Args:
        filepath (str): The filepath to the Excel file.
        sheet_name (str): The name of the sheet to load.

    Returns:
        tuple: A tuple containing the preprocessed features (X), labels (y), and the data transformer.
    """
    df = load_data(filepath, sheet_name)
    X, y, transformer = preprocess_data(df)
    return X, y, transformer


def build_and_train_model(X, y, transformer):
    """
    Build and train the churn prediction model.

    Args:
        X (numpy.ndarray): The preprocessed features.
        y (numpy.ndarray): The labels.
        transformer (sklearn.compose.ColumnTransformer): The data transformer.

    Returns:
        tuple: A tuple containing the trained model, the training and test sets, the predicted labels, and the model's accuracy.
    """
    model = build_model(transformer)
    model, X_train, y_train, X_test, y_test, y_pred, accuracy = train_model(model, X, y)
    return model, X_train, y_train, X_test, y_test, y_pred, accuracy


def tune_and_evaluate(model, X_train, y_train, X_test, y_test):
    """
    Tune the model's hyperparameters and evaluate its performance.

    Args:
        model (sklearn.base.BaseEstimator): The trained model.
        X_train (numpy.ndarray): The training features.
        y_train (numpy.ndarray): The training labels.
        X_test (numpy.ndarray): The test features.
        y_test (numpy.ndarray): The test labels.

    Returns:
        tuple: A tuple containing the tuned model, the predicted labels, accuracy, precision, recall, and F1-score.
    """
    grid_cv_model = parameter_tuning(model, X_train, y_train)
    y_pred = grid_cv_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    return grid_cv_model, y_pred, accuracy, precision, recall, f1


def log_metrics(accuracy, precision, recall, f1):
    """
    Log the model's performance metrics to MLflow.

    Args:
        accuracy (float): The model's accuracy.
        precision (float): The model's precision.
        recall (float): The model's recall.
        f1 (float): The model's F1-score.
    """
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)


def run_experiment():
    """
    Run the churn prediction experiment.
    """
    with mlflow.start_run() as run:
        try:
            print("Run ID:", run.info.run_id)
            X, y, transformer = load_and_preprocess_data("//Users/slaiby/Desktop/Methodo/E Commerce Dataset.xlsx", "E Comm")
            model, X_train, y_train, X_test, y_test, y_pred, accuracy = build_and_train_model(X, y, transformer)
            grid_cv_model, y_pred, accuracy, precision, recall, f1 = tune_and_evaluate(model, X_train, y_train, X_test, y_test)

            mlflow.sklearn.log_model(grid_cv_model, "model")
            log_metrics(accuracy, precision, recall, f1)

            print(f"Model accuracy: {accuracy * 100:.2f}%")
        except Exception as e:
            mlflow.log_param("error_message", str(e))
            raise

if __name__ == "__main__":
    run_experiment()