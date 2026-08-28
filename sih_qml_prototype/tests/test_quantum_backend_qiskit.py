import pytest
import numpy as np
from core.quantum_backend_qiskit import QiskitVariationalQuantumCircuit
from core.quantum_core import VariationalQuantumCircuit as NumpyVQC

def test_qiskit_forward_and_proba():
    vqc = QiskitVariationalQuantumCircuit(n_qubits=4, n_layers=2, seed=42)
    x = np.array([0.1, -0.2, 0.5, -0.3])
    score = vqc.forward(x)
    assert -1.0 <= score <= 1.0
    
    probs = vqc.predict_proba(np.array([x]))
    assert 0.0 <= probs[0] <= 1.0

def test_qiskit_numpy_mathematical_equivalence():
    x = np.array([0.25, -0.75, 0.5, -0.1])
    
    np_vqc = NumpyVQC(n_qubits=4, n_layers=2, seed=42)
    q_vqc = QiskitVariationalQuantumCircuit(n_qubits=4, n_layers=2, seed=42)
    
    # Set identical parameters
    theta = np.random.uniform(-0.5, 0.5, size=(2, 4, 2))
    np_vqc.params = theta.copy()
    q_vqc.params = theta.copy()
    
    np_val = np_vqc.forward(x)
    q_val = q_vqc.forward(x)
    
    assert abs(np_val - q_val) < 1e-10
