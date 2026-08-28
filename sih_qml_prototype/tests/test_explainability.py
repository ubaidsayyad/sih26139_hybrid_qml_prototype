import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression
from core.explainability import permutation_importance, shap_feature_importance

def test_permutation_importance():
    X = np.random.uniform(-1, 1, size=(60, 4))
    # Feature 0 determines label strongly
    y = (X[:, 0] > 0).astype(int)
    
    def predict_fn(X_sub):
        # High probability if feature 0 > 0
        return 1.0 / (1.0 + np.exp(-3.0 * X_sub[:, 0]))

    feature_names = ["PC1", "PC2", "PC3", "PC4"]
    baseline_auc, importances = permutation_importance(predict_fn, X, y, feature_names, n_repeats=5)
    
    assert 0.5 <= baseline_auc <= 1.0
    assert len(importances) == 4
    # PC1 should have the highest importance drop
    assert importances[0]["feature"] == "PC1"
    assert importances[0]["importance"] > 0.05

def test_shap_feature_importance():
    X_train = np.random.normal(0, 1, size=(80, 4))
    y_train = (X_train[:, 0] * 2 + X_train[:, 1] > 0).astype(int)
    X_test = np.random.normal(0, 1, size=(30, 4))
    
    model = LogisticRegression().fit(X_train, y_train)
    feature_names = ["PC1", "PC2", "PC3", "PC4"]
    
    _, importances = shap_feature_importance(model, X_train, X_test, feature_names)
    assert len(importances) == 4
    assert all("feature" in d and "importance" in d for d in importances)
    assert importances[0]["importance"] >= importances[-1]["importance"]
