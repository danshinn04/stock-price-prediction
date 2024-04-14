from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import numpy as np

# Linear Regression
def perform_linear_regression(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

# Ridge Regression
def perform_ridge_regression(X, y, alpha=1.0):
    model = Ridge(alpha=alpha)
    model.fit(X, y)
    return model

# Lasso Regression
def perform_lasso_regression(X, y, alpha=1.0):
    model = Lasso(alpha=alpha)
    model.fit(X, y)
    return model

# Elastic Net Regression
def perform_elasticnet_regression(X, y, alpha=1.0, l1_ratio=0.5):
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
    model.fit(X, y)
    return model

# Support Vector Regression
def perform_svr(X, y, kernel='rbf'):
    model = SVR(kernel=kernel)
    model.fit(X, y)
    return model

# Decision Tree Regression
def perform_decision_tree_regression(X, y):
    model = DecisionTreeRegressor()
    model.fit(X, y)
    return model

# Random Forest Regression
def perform_random_forest_regression(X, y, n_estimators=100):
    model = RandomForestRegressor(n_estimators=n_estimators)
    model.fit(X, y)
    return model

# Gradient Boosting Regression
def perform_gradient_boosting_regression(X, y, n_estimators=100):
    model = GradientBoostingRegressor(n_estimators=n_estimators)
    model.fit(X, y)
    return model

# Polynomial Regression
def perform_polynomial_regression(X, y, degree=2):
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X, y)
    return model
