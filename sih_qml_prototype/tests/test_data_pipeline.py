import pytest
import numpy as np
from core.data_pipeline import load_dataset, build_splits

def test_load_dataset():
    X, y, feature_names = load_dataset()
    assert X.shape == (569, 30)
    assert y.shape == (569,)
    assert set(np.unique(y)) == {0, 1}
    assert len(feature_names) == 30
    assert "mean radius" in feature_names

def test_build_splits_shapes():
    X, y, _ = load_dataset()
    splits = build_splits(X, y, n_qubits=4, test_size=0.25, train_subsample=100)
    
    # Classical 30-feature splits
    assert splits["classical"]["X_train"].shape[1] == 30
    assert splits["classical"]["X_test"].shape[1] == 30
    assert len(splits["classical"]["y_test"]) == int(np.ceil(569 * 0.25))

    # Quantum 4-qubit PCA splits
    assert splits["quantum"]["X_train"].shape == (100, 4)
    assert splits["quantum"]["X_test"].shape[1] == 4
    assert len(splits["quantum"]["y_train"]) == 100

    # Classical same features
    assert splits["classical_same_features"]["X_train"].shape[1] == 4
    assert splits["classical_same_features"]["X_test"].shape[1] == 4

def test_build_splits_stratification():
    X, y, _ = load_dataset()
    splits = build_splits(X, y, n_qubits=4, test_size=0.25)
    
    train_pos_rate = np.mean(splits["classical"]["y_train"])
    test_pos_rate = np.mean(splits["classical"]["y_test"])
    overall_pos_rate = np.mean(y)
    
    # Stratification ensures positive rate matches within 1%
    assert abs(train_pos_rate - overall_pos_rate) < 0.015
    assert abs(test_pos_rate - overall_pos_rate) < 0.015

def test_quantum_features_bounded():
    X, y, _ = load_dataset()
    splits = build_splits(X, y, n_qubits=4)
    
    # Min-max scaled for angle encoding in [-1, 1]
    assert np.all(splits["quantum"]["X_train"] >= -1.0 - 1e-6)
    assert np.all(splits["quantum"]["X_train"] <= 1.0 + 1e-6)
