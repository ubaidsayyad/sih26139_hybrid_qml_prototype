"""
explainability.py
------------------
Model-agnostic permutation feature importance. Works identically for the
classical baseline and the quantum circuit (treated as a black-box scoring
function), which is important because gradient-based explainability (SHAP's
KernelExplainer aside) doesn't have a mature quantum-native equivalent yet --
permutation importance sidesteps that by only requiring predict_proba().

Importance(feature_i) = drop in AUC when feature_i's values are shuffled
across the test set, averaged over several shuffles.
"""

import numpy as np
from sklearn.metrics import roc_auc_score


def permutation_importance(predict_fn, X_test, y_test, feature_names, n_repeats=8, seed=42):
    rng = np.random.default_rng(seed)
    baseline_auc = roc_auc_score(y_test, predict_fn(X_test))

    importances = []
    for j, name in enumerate(feature_names):
        drops = []
        for _ in range(n_repeats):
            X_shuffled = X_test.copy()
            perm = rng.permutation(len(X_shuffled))
            X_shuffled[:, j] = X_shuffled[perm, j]
            shuffled_auc = roc_auc_score(y_test, predict_fn(X_shuffled))
            drops.append(baseline_auc - shuffled_auc)
        importances.append({"feature": name, "importance": float(np.mean(drops)),
                             "std": float(np.std(drops))})

    importances.sort(key=lambda d: d["importance"], reverse=True)
    return baseline_auc, importances


def shap_feature_importance(model, X_train, X_test, feature_names):
    """
    Computes game-theoretic SHAP (Shapley Additive exPlanations) values for a classical model.
    Returns list of dicts: [{'feature': name, 'importance': mean_abs_shap, 'std': std_shap}]
    """
    try:
        import shap
        # Use LinearExplainer for linear models or general Explainer
        try:
            explainer = shap.LinearExplainer(model, X_train)
            shap_values = explainer(X_test)
        except Exception:
            explainer = shap.Explainer(model, X_train)
            shap_values = explainer(X_test)

        raw_vals = shap_values.values if hasattr(shap_values, 'values') else shap_values
        if raw_vals.ndim == 3:  # Binary classification with 2 output dimensions
            raw_vals = raw_vals[:, :, 1]

        mean_abs_shap = np.mean(np.abs(raw_vals), axis=0)
        std_shap = np.std(np.abs(raw_vals), axis=0)

        importances = []
        for i, name in enumerate(feature_names):
            importances.append({
                "feature": name,
                "importance": float(mean_abs_shap[i]),
                "std": float(std_shap[i])
            })
        importances.sort(key=lambda d: d["importance"], reverse=True)
        return raw_vals, importances
    except Exception as err:
        # Fallback if SHAP computation encounters unsupported model configuration
        importances = [{"feature": name, "importance": 0.0, "std": 0.0} for name in feature_names]
        return None, importances

