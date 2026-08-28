import pytest
import numpy as np
from sklearn.datasets import make_classification
from core.classical_baseline import train_and_eval

def test_logistic_regression_toy_data():
    X, y = make_classification(n_samples=200, n_features=10, n_informative=8, random_state=42)
    X_train, X_test = X[:150], X[150:]
    y_train, y_test = y[:150], y[150:]
    
    model, metrics, proba = train_and_eval(X_train, y_train, X_test, y_test, model_name="logistic_regression")
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "auc" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["auc"] <= 1.0
    assert len(proba) == len(y_test)
    assert np.all(proba >= 0.0) and np.all(proba <= 1.0)

def test_random_forest_toy_data():
    X, y = make_classification(n_samples=100, n_features=6, random_state=42)
    X_train, X_test = X[:70], X[70:]
    y_train, y_test = y[:70], y[70:]
    
    model, metrics, proba = train_and_eval(X_train, y_train, X_test, y_test, model_name="random_forest")
    assert metrics["accuracy"] > 0.5
    assert len(proba) == 30
