# linear_regression.py

from sklearn.linear_model import LinearRegression, Ridge
import numpy as np

def perform_linear_regression(X, y):
    """
    Performs linear regression on the provided data.
    
    Args:
    X (np.ndarray): The input features (independent variables).
    y (np.ndarray): The target values (dependent variable).

    Returns:
    LinearRegression: A fitted linear regression model.
    """
    model = LinearRegression()
    model.fit(X, y)
    return model

def perform_ridge_regression(X, y):
    model = Ridge()
    model.fit(X, y)
    return model