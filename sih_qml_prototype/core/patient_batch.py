"""
patient_batch.py
----------------
In-memory patient CSV parsing, strict schema validation, and multi-model inference.
Zero disk persistence -- processes stream entirely in RAM for patient privacy.
"""

import io
import re
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

# Canonical 30 biomedical feature names from Wisconsin Diagnostic dataset
FEATURE_NAMES = [str(f) for f in load_breast_cancer().feature_names]


def _normalize_col(name):
    """Normalize column header: lowercase, alphanumeric and single spaces only."""
    return re.sub(r'[\s_]+', ' ', str(name).strip().lower())


def generate_sample_csv(n_samples=10, include_ground_truth=True):
    """
    Generate a sample CSV in memory from the benchmark dataset with standard headers.
    Returns a CSV formatted string.
    """
    data = load_breast_cancer()
    X = data.data
    y = 1 - data.target  # 1 = Malignant, 0 = Benign

    # Select balanced subset
    rng = np.random.default_rng(42)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    
    n_pos = n_samples // 2
    n_neg = n_samples - n_pos
    
    chosen_pos = rng.choice(pos_idx, size=min(n_pos, len(pos_idx)), replace=False)
    chosen_neg = rng.choice(neg_idx, size=min(n_neg, len(neg_idx)), replace=False)
    
    sample_indices = np.concatenate([chosen_pos, chosen_neg])
    rng.shuffle(sample_indices)
    
    df = pd.DataFrame(X[sample_indices], columns=FEATURE_NAMES)
    df.insert(0, "patient_id", [f"PT-{1001 + i}" for i in range(len(sample_indices))])
    
    if include_ground_truth:
        df["ground_truth"] = ["Malignant" if y[i] == 1 else "Benign" for i in sample_indices]

    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def validate_and_process_csv(csv_content, run_data):
    """
    Validates patient CSV and executes quantum + classical models in-memory.
    
    Parameters:
        csv_content: str or bytes containing CSV text
        run_data: dict containing fitted models and scalers from run_full_pipeline
        
    Returns:
        tuple: (success: bool, data_or_error: dict)
    """
    if not run_data or "splits" not in run_data:
        return False, {"error": "Quantum pipeline has not been executed yet. Please run training first."}

    # 1. Parse CSV in-memory
    try:
        if isinstance(csv_content, bytes):
            csv_content = csv_content.decode('utf-8', errors='replace')
        
        df = pd.read_csv(io.StringIO(csv_content))
    except Exception as err:
        return False, {"error": f"Failed to parse CSV file: {str(err)}"}

    if df.empty:
        return False, {"error": "Uploaded CSV is completely empty (0 data rows found)."}

    # 2. Schema mapping
    df_cols = list(df.columns)
    normalized_df_cols = {_normalize_col(c): c for c in df_cols}
    
    # Check for patient ID column
    id_col = None
    for candidate in ["patient id", "patientid", "id", "sample id", "sampleid"]:
        if candidate in normalized_df_cols:
            id_col = normalized_df_cols[candidate]
            break

    # Check for optional ground truth label column
    gt_col = None
    for candidate in ["ground truth", "groundtruth", "diagnosis", "target", "label", "actual pathology", "class"]:
        if candidate in normalized_df_cols:
            gt_col = normalized_df_cols[candidate]
            break

    # 3. Match 30 required feature columns
    missing_features = []
    matched_col_map = {}  # canonical -> actual df column name

    for feat in FEATURE_NAMES:
        norm_feat = _normalize_col(feat)
        if norm_feat in normalized_df_cols:
            matched_col_map[feat] = normalized_df_cols[norm_feat]
        else:
            missing_features.append(feat)

    if missing_features:
        sample_missing = missing_features[:5]
        suffix = f" ... and {len(missing_features) - 5} more" if len(missing_features) > 5 else ""
        return False, {
            "error": f"Missing {len(missing_features)} required biomedical feature column(s): {sample_missing}{suffix}.",
            "missing_columns": missing_features,
            "expected_columns": FEATURE_NAMES
        }

    # 4. Validate numeric types and check for NaN/inf
    actual_feature_cols = [matched_col_map[f] for f in FEATURE_NAMES]
    feature_df = df[actual_feature_cols].copy()

    # Check for non-numeric entries
    for col in actual_feature_cols:
        coerced = pd.to_numeric(feature_df[col], errors='coerce')
        nan_mask = coerced.isna()
        if nan_mask.any():
            bad_row_idx = int(nan_mask.idxmax()) + 1
            bad_val = df.loc[nan_mask.idxmax(), col]
            return False, {
                "error": f"Invalid non-numeric or blank value '{bad_val}' found in column '{col}' at row {bad_row_idx}."
            }

    # Extract raw numeric matrix
    X_raw = feature_df.to_numpy(dtype=float)

    # 5. Extract ground truth if available
    ground_truths = []
    if gt_col is not None:
        for val in df[gt_col]:
            val_str = str(val).strip().lower()
            if val_str in ["1", "1.0", "malignant", "m", "positive", "pos", "true"]:
                ground_truths.append(1)
            elif val_str in ["0", "0.0", "benign", "b", "negative", "neg", "false"]:
                ground_truths.append(0)
            else:
                ground_truths.append(None)
    else:
        ground_truths = [None] * len(df)

    # 6. Extract patient IDs
    patient_ids = []
    for i in range(len(df)):
        if id_col is not None:
            patient_ids.append(str(df.loc[i, id_col]))
        else:
            patient_ids.append(f"Row #{i+1}")

    # 7. Apply saved preprocessing pipeline
    scaler = run_data["splits"]["scaler"]
    pca = run_data["splits"]["pca"]
    minmax = run_data["splits"]["minmax"]

    Xc = scaler.transform(X_raw)
    Xpca = pca.transform(Xc)
    Xq = minmax.transform(Xpca)

    # 8. Run Model Inferences
    vqc = run_data["vqc"]
    fair_model = run_data.get("fair_model")
    lr_model = run_data.get("lr_model")

    # Quantum VQC
    q_probs = vqc.predict_proba(Xq)

    # Classical Fair Baseline (same 4 PCA features)
    if fair_model is not None:
        c_probs = fair_model.predict_proba(Xq)[:, 1]
    else:
        c_probs = [0.5] * len(df)

    # Classical Full 30-feature Model
    if lr_model is not None:
        c30_probs = lr_model.predict_proba(Xc)[:, 1]
    else:
        c30_probs = [0.5] * len(df)

    # 9. Format patient-level results
    results_rows = []
    agree_count = 0
    q_correct_count = 0
    c_correct_count = 0
    evaluable_gt_count = 0

    for i in range(len(df)):
        qp = float(q_probs[i])
        cp = float(c_probs[i])
        c30p = float(c30_probs[i])
        gt = ground_truths[i]

        q_pred_int = 1 if qp >= 0.5 else 0
        c_pred_int = 1 if cp >= 0.5 else 0
        agrees = bool(q_pred_int == c_pred_int)
        if agrees:
            agree_count += 1

        q_conf = qp if q_pred_int == 1 else 1.0 - qp
        c_conf = cp if c_pred_int == 1 else 1.0 - cp

        q_correct = None
        c_correct = None
        if gt is not None:
            evaluable_gt_count += 1
            q_correct = bool(q_pred_int == gt)
            c_correct = bool(c_pred_int == gt)
            if q_correct:
                q_correct_count += 1
            if c_correct:
                c_correct_count += 1

        results_rows.append({
            "row_idx": i + 1,
            "patient_id": patient_ids[i],
            "quantum_prediction": "Malignant" if q_pred_int == 1 else "Benign",
            "quantum_prob": round(qp, 3),
            "quantum_conf": round(q_conf, 3),
            "classical_prediction": "Malignant" if c_pred_int == 1 else "Benign",
            "classical_prob": round(cp, 3),
            "classical_conf": round(c_conf, 3),
            "classical_30_prob": round(c30p, 3),
            "agreement": agrees,
            "ground_truth": "Malignant" if gt == 1 else ("Benign" if gt == 0 else None),
            "quantum_correct": q_correct,
            "classical_correct": c_correct,
        })

    total_samples = len(df)
    summary = {
        "total_samples": total_samples,
        "quantum_malignant_count": sum(1 for r in results_rows if r["quantum_prediction"] == "Malignant"),
        "classical_malignant_count": sum(1 for r in results_rows if r["classical_prediction"] == "Malignant"),
        "agreement_count": agree_count,
        "agreement_rate": round(agree_count / total_samples * 100, 1),
        "has_ground_truth": evaluable_gt_count > 0,
        "quantum_batch_accuracy": round(q_correct_count / evaluable_gt_count * 100, 1) if evaluable_gt_count > 0 else None,
        "classical_batch_accuracy": round(c_correct_count / evaluable_gt_count * 100, 1) if evaluable_gt_count > 0 else None,
        "quantum_backend": run_data.get("quantum_backend_display", "Local Quantum Simulator")
    }

    return True, {
        "summary": summary,
        "rows": results_rows
    }
