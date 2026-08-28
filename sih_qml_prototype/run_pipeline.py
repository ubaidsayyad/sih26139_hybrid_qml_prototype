"""
run_pipeline.py
----------------
End-to-end run of the Hybrid Quantum-Classical ML platform prototype
(SIH26139) on the Breast Cancer Wisconsin dataset:

  1. Data ingestion
  2. Classical pre-processing / feature engineering (scaling + PCA)
  3. Classical baselines (Logistic Regression, Random Forest)
  4. Hybrid Quantum Model (Variational Quantum Classifier, from-scratch
     numpy statevector simulator, trained via the parameter-shift rule)
  5. Evaluation: quantum vs classical, on equal footing (same PCA features)
  6. Explainability: permutation importance for every model

Run:  python3 run_pipeline.py
"""

import json
import time
import argparse
import numpy as np

from core.data_pipeline import load_dataset, build_splits
from core.classical_baseline import train_and_eval
from core.quantum_core import VariationalQuantumCircuit
from core.explainability import permutation_importance


def main():
    parser = argparse.ArgumentParser(description="SIH26139 Hybrid QML Pipeline Runner")
    parser.add_argument("--epochs", type=int, default=45, help="Number of quantum optimization epochs")
    parser.add_argument("--lr", type=float, default=0.4, help="Learning rate")
    parser.add_argument("--subsample", type=int, default=140, help="Training set subsample size")
    parser.add_argument("--n_qubits", type=int, default=4, help="Number of qubits / PCA dimensions")
    parser.add_argument("--n_layers", type=int, default=2, help="Number of variational circuit layers")
    args = parser.parse_args()

    N_QUBITS = args.n_qubits
    N_LAYERS = args.n_layers
    EPOCHS = args.epochs
    LR = args.lr

    print("=" * 70)
    print("SIH26139 -- Hybrid Quantum ML Platform for Early Disease Detection")
    print("Dataset: Breast Cancer Wisconsin (Diagnostic)  |  Task: malignant vs benign")
    print("=" * 70)

    # 1. Ingestion
    X, y, feature_names = load_dataset()
    print(f"\n[1/5] Loaded {X.shape[0]} samples, {X.shape[1]} raw features. "
          f"Positive (malignant) rate: {y.mean():.2%}")

    # 2. Preprocessing
    splits = build_splits(X, y, n_qubits=N_QUBITS, train_subsample=args.subsample)
    print(f"[2/5] Preprocessed: classical branch = {splits['classical']['X_train'].shape[1]} features "
          f"(standardized); quantum branch = {N_QUBITS} features (PCA + angle-encoded to [-1,1])")

    results = {}

    # 3. Classical baselines (full feature set)
    print("\n[3/5] Training classical baselines (full 30-feature set)...")
    for name in ["logistic_regression", "random_forest"]:
        model, metrics, proba = train_and_eval(
            splits["classical"]["X_train"], splits["classical"]["y_train"],
            splits["classical"]["X_test"], splits["classical"]["y_test"], name)
        results[name] = metrics
        print(f"   {name:22s}  acc={metrics['accuracy']:.3f}  f1={metrics['f1']:.3f}  "
              f"auc={metrics['auc']:.3f}  train_time={metrics['train_time_sec']:.3f}s")

    # 3b. Classical baseline on the SAME reduced features as the quantum model (fair fight)
    model_fair, metrics_fair, proba_fair = train_and_eval(
        splits["classical_same_features"]["X_train"], splits["classical_same_features"]["y_train"],
        splits["classical_same_features"]["X_test"], splits["classical_same_features"]["y_test"],
        "logistic_regression")
    results["logistic_regression_same_4_features"] = metrics_fair
    print(f"   {'logreg (4 PCA feats)':22s}  acc={metrics_fair['accuracy']:.3f}  "
          f"f1={metrics_fair['f1']:.3f}  auc={metrics_fair['auc']:.3f}  "
          f"train_time={metrics_fair['train_time_sec']:.3f}s")

    # 4. Hybrid Quantum Model
    print(f"\n[4/5] Training Variational Quantum Classifier "
          f"({N_QUBITS} qubits, {N_LAYERS} layers, {EPOCHS} epochs, parameter-shift gradients)...")
    vqc = VariationalQuantumCircuit(n_qubits=N_QUBITS, n_layers=N_LAYERS, seed=42)
    print(f"   Trainable quantum parameters: {vqc.n_params()}")

    t0 = time.time()
    history = vqc.fit(
        splits["quantum"]["X_train"], splits["quantum"]["y_train"],
        epochs=EPOCHS, lr=LR, verbose=10
    )
    q_train_time = time.time() - t0

    q_test_proba = vqc.predict_proba(splits["quantum"]["X_test"])
    q_test_pred = (q_test_proba >= 0.5).astype(int)
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    y_test_q = splits["quantum"]["y_test"]
    q_metrics = {
        "accuracy": accuracy_score(y_test_q, q_test_pred),
        "precision": precision_score(y_test_q, q_test_pred, zero_division=0),
        "recall": recall_score(y_test_q, q_test_pred, zero_division=0),
        "f1": f1_score(y_test_q, q_test_pred, zero_division=0),
        "auc": roc_auc_score(y_test_q, q_test_proba),
        "train_time_sec": q_train_time,
        "n_params": vqc.n_params(),
        "n_qubits": N_QUBITS,
        "loss_history": history,
    }
    results["hybrid_vqc"] = q_metrics
    print(f"   {'hybrid_vqc':22s}  acc={q_metrics['accuracy']:.3f}  f1={q_metrics['f1']:.3f}  "
          f"auc={q_metrics['auc']:.3f}  train_time={q_metrics['train_time_sec']:.3f}s")

    # 5. Explainability
    print("\n[5/5] Computing permutation importance (quantum model, black-box)...")
    pca_feature_names = [f"PC{i+1}" for i in range(N_QUBITS)]

    def quantum_predict(Xb):
        return vqc.predict_proba(Xb)

    base_auc, q_importances = permutation_importance(
        quantum_predict, splits["quantum"]["X_test"], y_test_q, pca_feature_names)
    results["quantum_explainability"] = {
        "baseline_auc": base_auc,
        "importances": q_importances,
    }
    for imp in q_importances:
        print(f"   {imp['feature']:6s}  importance(delta_AUC)={imp['importance']:+.4f}  "
              f"(std={imp['std']:.4f})")

    def classical_fair_predict(Xb):
        return model_fair.predict_proba(Xb)[:, 1]

    base_auc_c, c_importances = permutation_importance(
        classical_fair_predict, splits["classical_same_features"]["X_test"], y_test_q, pca_feature_names)
    results["classical_explainability"] = {
        "baseline_auc": base_auc_c,
        "importances": c_importances,
    }

    # Classical SHAP explainability
    from core.explainability import shap_feature_importance
    _, c_shap = shap_feature_importance(
        model_fair, splits["classical_same_features"]["X_train"],
        splits["classical_same_features"]["X_test"], pca_feature_names)
    results["classical_shap_explainability"] = {
        "importances": c_shap
    }

    import os
    os.makedirs("outputs", exist_ok=True)
    # Save results for the web dashboard
    with open("outputs/results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

    print("\n" + "=" * 70)
    print("SUMMARY  (Test set, held out, never seen during training)")
    print("=" * 70)
    header = f"{'Model':30s}{'Accuracy':>10s}{'F1':>8s}{'AUC':>8s}{'Train(s)':>10s}"
    print(header)
    print("-" * len(header))
    for name, key in [("Logistic Regression (30 feat)", "logistic_regression"),
                       ("Random Forest (30 feat)", "random_forest"),
                       ("Logistic Regression (4 PCA feat)", "logistic_regression_same_4_features"),
                       ("Hybrid VQC (4 qubits)", "hybrid_vqc")]:
        m = results[key]
        print(f"{name:30s}{m['accuracy']*100:9.1f}%{m['f1']:8.3f}{m['auc']:8.3f}{m['train_time_sec']:10.3f}")
    print("\nSaved full results -> outputs/results.json")


if __name__ == "__main__":
    main()
