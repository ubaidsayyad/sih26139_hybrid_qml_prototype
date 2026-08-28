import pytest
import numpy as np
from core.quantum_core import VariationalQuantumCircuit

def test_forward_bounds():
    vqc = VariationalQuantumCircuit(n_qubits=4, n_layers=2, seed=42)
    rng = np.random.default_rng(42)
    
    for _ in range(20):
        x = rng.uniform(-1, 1, size=4)
        score = vqc.forward(x)
        # Average Pauli-Z expectation value must be strictly in [-1, 1]
        assert -1.0 <= score <= 1.0

def test_predict_proba_bounds():
    vqc = VariationalQuantumCircuit(n_qubits=4, n_layers=2, seed=42)
    X = np.random.uniform(-1, 1, size=(10, 4))
    probs = vqc.predict_proba(X)
    assert len(probs) == 10
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)

def test_parameter_shift_gradient_shape():
    vqc = VariationalQuantumCircuit(n_qubits=4, n_layers=2, seed=42)
    X = np.random.uniform(-1, 1, size=(5, 4))
    y = np.array([1, 0, 1, 0, 1])
    
    grad = vqc.parameter_shift_gradients(X, y)
    assert grad.shape == vqc.params.shape
    assert grad.shape == (2, 4, 2)
    assert not np.isnan(grad).any()

def test_fit_reduces_loss():
    vqc = VariationalQuantumCircuit(n_qubits=4, n_layers=2, seed=42)
    X = np.random.uniform(-1, 1, size=(10, 4))
    y = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    
    initial_loss = vqc.compute_loss(X, y)
    history = vqc.fit(X, y, epochs=4, lr=0.5)
    final_loss = history[-1]
    assert final_loss < initial_loss
