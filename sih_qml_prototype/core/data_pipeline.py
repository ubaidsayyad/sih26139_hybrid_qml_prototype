"""
data_pipeline.py
-----------------
Data ingestion + classical pre-processing / feature engineering stage.

For the prototype we use the Breast Cancer Wisconsin (Diagnostic) dataset
(built into scikit-learn, so it works fully offline) as the demo biomedical
dataset. In production this module would instead ingest genomics / EHR /
imaging-derived tabular features from a hospital data source or public
dataset (e.g. MIMIC-III, TCGA) -- the rest of the platform is dataset-agnostic
as long as it receives a numeric feature matrix + binary labels.
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def load_dataset():
    data = load_breast_cancer()
    X, y = data.data, data.target
    # sklearn encodes 0=malignant, 1=benign; flip so 1 = disease-positive (malignant)
    y = 1 - y
    return X, y, list(data.feature_names)


def build_splits(X, y, n_qubits=4, test_size=0.25, train_subsample=140, seed=42):
    """
    Returns a dict with:
      - classical: full-dimensional standardized train/test split (for the
        classical baseline, which can use all 30 features)
      - quantum:   PCA-reduced (n_qubits-dim) + min-max scaled to [-1,1]
        train/test split (angle-encoding needs a bounded, low-dim input)
      - pca: fitted PCA object (used later for explainability)
    """
    X_train_full, X_test_full, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    # ---- classical branch: standardized, all features ----
    scaler = StandardScaler().fit(X_train_full)
    Xc_train = scaler.transform(X_train_full)
    Xc_test = scaler.transform(X_test_full)

    # ---- quantum branch: PCA to n_qubits dims, scaled to [-1, 1] for angle encoding ----
    pca = PCA(n_components=n_qubits, random_state=seed).fit(Xc_train)
    Xq_train = pca.transform(Xc_train)
    Xq_test = pca.transform(Xc_test)

    mm = MinMaxScaler(feature_range=(-1, 1)).fit(Xq_train)
    Xq_train = mm.transform(Xq_train)
    Xq_test = mm.transform(Xq_test)

    # Subsample the quantum training set for tractable simulator runtime
    # (parameter-shift training scales with #samples x #params x 2 forward passes/epoch)
    rng = np.random.default_rng(seed)
    if train_subsample and train_subsample < len(Xq_train):
        idx = rng.choice(len(Xq_train), size=train_subsample, replace=False)
        Xq_train_sub, y_train_sub = Xq_train[idx], y_train[idx]
    else:
        Xq_train_sub, y_train_sub = Xq_train, y_train

    return {
        "classical": {
            "X_train": Xc_train, "X_test": Xc_test,
            "y_train": y_train, "y_test": y_test,
        },
        "classical_same_features": {
            # classical baseline trained on the SAME PCA-reduced features the
            # quantum model sees -- an apples-to-apples comparison, isolating
            # whether the quantum circuit adds value beyond the info bottleneck
            "X_train": Xq_train, "X_test": Xq_test,
            "y_train": y_train, "y_test": y_test,
        },
        "quantum": {
            "X_train": Xq_train_sub, "X_test": Xq_test,
            "y_train": y_train_sub, "y_test": y_test,
        },
        "pca": pca,
        "scaler": scaler,
        "minmax": mm,
    }
