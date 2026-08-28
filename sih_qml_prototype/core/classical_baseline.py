"""
classical_baseline.py
----------------------
Purely classical models used as the benchmark the hybrid quantum model is
evaluated against, as required by the problem statement.
"""

import time
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def train_and_eval(X_train, y_train, X_test, y_test, model_name="logistic_regression"):
    if model_name == "logistic_regression":
        model = LogisticRegression(max_iter=2000)
    elif model_name == "random_forest":
        model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
    else:
        raise ValueError(model_name)

    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "auc": roc_auc_score(y_test, y_proba),
        "train_time_sec": train_time,
    }
    return model, metrics, y_proba
