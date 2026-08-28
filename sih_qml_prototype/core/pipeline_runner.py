"""
pipeline_runner.py
-------------------
Callable version of the end-to-end pipeline (ingestion -> preprocessing ->
classical baselines -> hybrid quantum model -> explainability) that returns
everything the Flask dashboard needs to render charts and tables.
"""
import time
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from core.data_pipeline import load_dataset, build_splits
from core.classical_baseline import train_and_eval
from core.quantum_core import VariationalQuantumCircuit as NumpyVariationalQuantumCircuit
from core.quantum_backend_qiskit import QiskitVariationalQuantumCircuit
from core.explainability import permutation_importance, shap_feature_importance


def run_full_pipeline(n_qubits=4, n_layers=2, epochs=30, lr=0.4, train_subsample=140,
                      quantum_backend="simulator", ibm_token=None, log=None, progress_callback=None):
    def _log(msg, pct=None):
        if log:
            log(msg)
        if progress_callback and pct is not None:
            progress_callback(pct, msg)

    backend_labels = {
        "simulator": "Local NumPy Simulator (Offline)",
        "qiskit_simulator": "Qiskit Statevector Simulator",
        "ibm_hardware": "IBM Quantum Hardware (QPU)"
    }
    backend_display = backend_labels.get(quantum_backend, quantum_backend)

    _log(f"Initializing pipeline with backend: {backend_display}...", 2)
    _log("Ingesting biomedical dataset & generating stratified partitions...", 5)
    X, y, feature_names = load_dataset()
    _log(f"Loaded {X.shape[0]} samples, {X.shape[1]} raw features (positive rate {y.mean():.1%})", 10)

    splits = build_splits(X, y, n_qubits=n_qubits, train_subsample=train_subsample)
    _log(f"Preprocessed: classical=30 std. features | quantum={n_qubits} PCA features, angle-encoded", 15)

    results = {
        "quantum_backend": quantum_backend,
        "quantum_backend_display": backend_display
    }
    probas = {}

    _log("Training Logistic Regression (30 features)...", 18)
    lr_model, lr_metrics, lr_proba = train_and_eval(
        splits["classical"]["X_train"], splits["classical"]["y_train"],
        splits["classical"]["X_test"], splits["classical"]["y_test"], "logistic_regression")
    results["logistic_regression"] = lr_metrics
    probas["Logistic Regression"] = (splits["classical"]["y_test"], lr_proba)

    _log("Training Random Forest (30 features, 200 trees)...", 22)
    rf_model, rf_metrics, rf_proba = train_and_eval(
        splits["classical"]["X_train"], splits["classical"]["y_train"],
        splits["classical"]["X_test"], splits["classical"]["y_test"], "random_forest")
    results["random_forest"] = rf_metrics

    _log("Training Logistic Regression on same 4 PCA features (fair baseline)...", 26)
    fair_model, fair_metrics, fair_proba = train_and_eval(
        splits["classical_same_features"]["X_train"], splits["classical_same_features"]["y_train"],
        splits["classical_same_features"]["X_test"], splits["classical_same_features"]["y_test"],
        "logistic_regression")
    results["logistic_regression_same_4_features"] = fair_metrics

    _log(f"Initializing {backend_display} ({n_qubits} qubits, {n_layers} layers)...", 30)
    
    if quantum_backend == "simulator":
        vqc = NumpyVariationalQuantumCircuit(n_qubits=n_qubits, n_layers=n_layers, seed=42)
    elif quantum_backend in ("qiskit_simulator", "ibm_hardware"):
        vqc = QiskitVariationalQuantumCircuit(
            n_qubits=n_qubits,
            n_layers=n_layers,
            seed=42,
            backend_mode=quantum_backend,
            ibm_token=ibm_token
        )
    else:
        raise ValueError(f"Unknown quantum backend: {quantum_backend}")
    
    def _on_epoch(epoch, total_epochs, loss):
        pct = 30 + int(50 * (epoch / total_epochs))
        msg = f"Training Quantum Circuit ({backend_display}) — Epoch {epoch}/{total_epochs} (Loss: {loss:.4f})"
        _log(msg, pct)

    t0 = time.time()
    history = vqc.fit(splits["quantum"]["X_train"], splits["quantum"]["y_train"],
                       epochs=epochs, lr=lr,
                       verbose=max(1, epochs // 5) if log else None,
                       epoch_callback=_on_epoch)
    q_train_time = time.time() - t0

    _log("Evaluating quantum test predictions...", 82)
    y_test_q = splits["quantum"]["y_test"]
    q_proba = vqc.predict_proba(splits["quantum"]["X_test"])
    q_pred = (q_proba >= 0.5).astype(int)
    q_metrics = {
        "accuracy": accuracy_score(y_test_q, q_pred),
        "precision": precision_score(y_test_q, q_pred, zero_division=0),
        "recall": recall_score(y_test_q, q_pred, zero_division=0),
        "f1": f1_score(y_test_q, q_pred, zero_division=0),
        "auc": roc_auc_score(y_test_q, q_proba),
        "train_time_sec": q_train_time,
        "n_params": vqc.n_params(),
        "n_qubits": n_qubits,
        "n_layers": n_layers,
        "backend": backend_display,
        "loss_history": history,
    }
    results["hybrid_vqc"] = q_metrics
    probas["Hybrid VQC"] = (y_test_q, q_proba)
    _log(f"Quantum model evaluated: Acc={q_metrics['accuracy']:.1%}, AUC={q_metrics['auc']:.3f}", 86)

    _log("Computing permutation feature importance (Hybrid VQC)...", 88)
    pca_names = [f"PC{i+1}" for i in range(n_qubits)]
    _, q_importances = permutation_importance(
        lambda Xb: vqc.predict_proba(Xb), splits["quantum"]["X_test"], y_test_q, pca_names)
    results["quantum_explainability"] = {"importances": q_importances}

    _log("Computing permutation feature importance (Classical Fair Baseline)...", 92)
    _, c_importances = permutation_importance(
        lambda Xb: fair_model.predict_proba(Xb)[:, 1],
        splits["classical_same_features"]["X_test"], y_test_q, pca_names)
    results["classical_explainability"] = {"importances": c_importances}

    _log("Computing SHAP feature attribution (Classical Fair Baseline)...", 96)
    _, c_shap_importances = shap_feature_importance(
        fair_model,
        splits["classical_same_features"]["X_train"],
        splits["classical_same_features"]["X_test"],
        pca_names)
    results["classical_shap_explainability"] = {"importances": c_shap_importances}

    _log("Pipeline run complete.", 100)
    return {
        "results": results,
        "probas": probas,
        "loss_history": history,
        "vqc": vqc,
        "fair_model": fair_model,
        "lr_model": lr_model,
        "rf_model": rf_model,
        "splits": splits,
        "feature_names": feature_names,
        "quantum_backend": quantum_backend,
        "quantum_backend_display": backend_display,
    }
