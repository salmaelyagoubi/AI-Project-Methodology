"""
This module contains functions for loading and preprocessing the data for the churn prediction model.
"""
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def load_data(filepath, sheetname):
    """
    Load the data from an Excel file.

    Args:
        filepath (str): The filepath to the Excel file.
        sheetname (str): The name of the sheet to load.

    Returns:
        pandas.DataFrame: The loaded data.
    """
    df = pd.read_excel(filepath, sheet_name=sheetname)
    df.drop(columns="CustomerID", inplace=True)
    return df


def preprocess_data(df):
    """
    Preprocess the data for the churn prediction model.

    Args:
        df (pandas.DataFrame): The input data.

    Returns:
        tuple: A tuple containing the preprocessed features (X), labels (y), and the data transformer.
    """
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    cat_cols = X.select_dtypes(include="O").columns
    num_cols = X.columns.difference(cat_cols)

    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy="most_frequent")),
        ('encoding', OneHotEncoder())
    ])

    numerical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy="mean")),
        ('scaler', StandardScaler())
    ])

    transformer = ColumnTransformer(transformers=[
        ('cat', categorical_pipeline, cat_cols),
        ('num', numerical_pipeline, num_cols)
    ])
    
    return X, y, transformer