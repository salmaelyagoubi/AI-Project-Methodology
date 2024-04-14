"""
This module contains functions for building, training, and tuning the churn prediction model.
"""
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline


def build_model(transformer):
    """
    Build the churn prediction model.

    Args:
        transformer (sklearn.compose.ColumnTransformer): The data transformer.

    Returns:
        sklearn.pipeline.Pipeline: The model pipeline.
    """
    classifier = XGBClassifier()
    model = Pipeline([
        ('transformer', transformer),
        ('classifier', classifier)
    ])
    return model


def train_model(model, X, y):
    """
    Train the churn prediction model.

    Args:
        model (sklearn.pipeline.Pipeline): The model pipeline.
        X (numpy.ndarray): The features.
        y (numpy.ndarray): The labels.

    Returns:
        tuple: A tuple containing the trained model, the training and test sets, the predicted labels, and the model's accuracy.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return model, X_train, y_train, X_test, y_test, y_pred, accuracy


def parameter_tuning(model, X_train, y_train):
    """
    Tune the model's hyperparameters using GridSearchCV.

    Args:
        model (sklearn.pipeline.Pipeline): The model pipeline.
        X_train (numpy.ndarray): The training features.
        y_train (numpy.ndarray): The training labels.

    Returns:
        sklearn.model_selection.GridSearchCV: The tuned model.
    """
    param_grid = {
        'classifier__n_estimators': [350],
        'classifier__learning_rate': [0.09],
        'classifier__max_depth': [25],
        'classifier__min_child_weight': [1],
        'classifier__subsample': [0.9],
        'classifier__colsample_bytree': [1.0],
        'classifier__lambda': [1]
    }
    grid_cv = GridSearchCV(model, param_grid, scoring='accuracy', cv=5, verbose=1, n_jobs=-1)
    grid_cv.fit(X_train, y_train)
    return grid_cv


def log_experiment(model, accuracy):
    """
    Log the experiment details to MLflow.

    Args:
        model (sklearn.base.BaseEstimator): The trained model.
        accuracy (float): The model's accuracy.
    """
    mlflow.log_param("model_type", type(model).__name__)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "model")