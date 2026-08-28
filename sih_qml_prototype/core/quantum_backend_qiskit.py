"""
quantum_backend_qiskit.py
-------------------------
Qiskit-based drop-in replacement for the core.quantum_core.VariationalQuantumCircuit.

Supports two operational modes:
1. "qiskit_simulator" : Local statevector simulation using Qiskit quantum_info primitives
2. "ibm_hardware"     : Real quantum hardware execution via Qiskit IBM Runtime (EstimatorV2 / QPU)

Implements the exact same interface:
- forward(x, params=None) -> score in [-1, 1]
- predict_proba(X, params=None) -> probs in [0, 1]
- compute_loss(X, y, params=None) -> BCE loss
- parameter_shift_gradients(X, y) -> gradient array
- fit(X, y, epochs, lr, verbose, epoch_callback) -> loss history
"""

import os
import time
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector, SparsePauliOp


class QiskitVariationalQuantumCircuit:
    """
    Qiskit-native implementation of the Hardware-Efficient Variational Quantum Classifier.
    """

    def __init__(self, n_qubits=4, n_layers=2, seed=42, backend_mode="qiskit_simulator",
                 ibm_token=None, ibm_instance=None):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.backend_mode = backend_mode
        self.ibm_token = ibm_token or os.environ.get("IBM_QUANTUM_TOKEN") or os.environ.get("QISKIT_IBM_TOKEN")
        self.ibm_instance = ibm_instance
        self.ibm_service = None
        self.ibm_backend = None

        rng = np.random.default_rng(seed)
        # Parameter tensor shape: (n_layers, n_qubits, 2)
        self.params = rng.uniform(-0.1, 0.1, size=(n_layers, n_qubits, 2))

        # Build parameterized Qiskit circuit template
        self.x_params = ParameterVector('x', self.n_qubits)
        self.theta_params = ParameterVector('theta', self.n_layers * self.n_qubits * 2)
        self.circuit = self._build_circuit_template()

        # Build observable: average <Z> across all qubits
        # In Qiskit, qubit 0 is the rightmost index in the Pauli string
        pauli_terms = []
        for q in range(self.n_qubits):
            z_str = ['I'] * self.n_qubits
            z_str[self.n_qubits - 1 - q] = 'Z'
            pauli_terms.append((''.join(z_str), 1.0 / self.n_qubits))
        self.observable = SparsePauliOp.from_list(pauli_terms)

        # Initialize IBM Quantum runtime if requested
        if self.backend_mode == "ibm_hardware":
            self._init_ibm_runtime()

    def _init_ibm_runtime(self):
        if not self.ibm_token:
            raise ValueError(
                "IBM Quantum API token not found! To execute on real quantum hardware, "
                "set the environment variable IBM_QUANTUM_TOKEN=<your_api_token> "
                "or provide it in the backend configuration."
            )
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService
            try:
                self.ibm_service = QiskitRuntimeService(channel="ibm_quantum_platform", token=self.ibm_token)
            except Exception:
                try:
                    self.ibm_service = QiskitRuntimeService(channel="ibm_quantum", token=self.ibm_token)
                except Exception:
                    self.ibm_service = QiskitRuntimeService(channel="ibm_cloud", token=self.ibm_token)
            
            # Find least busy operational QPU with enough qubits
            self.ibm_backend = self.ibm_service.least_busy(
                operational=True,
                simulator=False,
                min_num_qubits=self.n_qubits
            )
            print(f"[Qiskit Backend] Connected to IBM Quantum hardware: {self.ibm_backend.name}")
        except Exception as err:
            raise RuntimeError(f"Failed to initialize IBM Quantum runtime service: {str(err)}")

    def _build_circuit_template(self):
        qc = QuantumCircuit(self.n_qubits)
        
        # 1. Feature Encoding Layer: RY(pi * x_i)
        for q in range(self.n_qubits):
            qc.ry(np.pi * self.x_params[q], q)

        # 2. Variational Layers
        p_idx = 0
        for l in range(self.n_layers):
            for q in range(self.n_qubits):
                qc.ry(self.theta_params[p_idx], q)
                p_idx += 1
                qc.rz(self.theta_params[p_idx], q)
                p_idx += 1
            # Ring CNOT Entanglement
            for q in range(self.n_qubits):
                qc.cx(q, (q + 1) % self.n_qubits)

        return qc

    def n_params(self):
        return self.params.size

    def _bind_circuit(self, x, flat_theta):
        param_dict = {}
        for i in range(self.n_qubits):
            param_dict[self.x_params[i]] = float(x[i])
        for j in range(len(flat_theta)):
            param_dict[self.theta_params[j]] = float(flat_theta[j])
        return self.circuit.assign_parameters(param_dict)

    def forward(self, x, params=None):
        """Run the circuit for one sample x. Returns score in [-1, 1] (average <Z>)."""
        if params is None:
            params = self.params
        flat_theta = params.reshape(-1)

        if self.backend_mode == "ibm_hardware" and self.ibm_backend is not None:
            # Qiskit Runtime Estimator on real hardware
            from qiskit_ibm_runtime import EstimatorV2
            from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
            
            pm = generate_preset_pass_manager(backend=self.ibm_backend, optimization_level=1)
            bound_qc = self._bind_circuit(x, flat_theta)
            isa_circuit = pm.run(bound_qc)
            isa_obs = self.observable.apply_layout(isa_circuit.layout)
            
            estimator = EstimatorV2(mode=self.ibm_backend)
            job = estimator.run([(isa_circuit, isa_obs)])
            pub_result = job.result()[0]
            return float(pub_result.data.evs)
        else:
            # Local Qiskit Statevector simulation
            bound_qc = self._bind_circuit(x, flat_theta)
            sv = Statevector.from_instruction(bound_qc)
            exp_val = sv.expectation_value(self.observable).real
            return float(exp_val)

    def predict_proba(self, X, params=None):
        """Map expectation values from [-1, 1] to binary probability [0, 1]."""
        scores = np.array([self.forward(x, params) for x in X])
        return (scores + 1.0) / 2.0

    def compute_loss(self, X, y, params=None):
        probs = self.predict_proba(X, params)
        eps = 1e-7
        probs = np.clip(probs, eps, 1 - eps)
        return float(-np.mean(y * np.log(probs) + (1 - y) * np.log(1 - probs)))

    def parameter_shift_gradients(self, X, y):
        """Analytic parameter-shift rule on Qiskit circuit:
        dE/dtheta = [E(theta + pi/2) - E(theta - pi/2)] / 2
        """
        shift = np.pi / 2
        grad = np.zeros_like(self.params)
        base_probs = self.predict_proba(X)
        dL_dprob = (base_probs - y) / (base_probs * (1 - base_probs) + 1e-7) / len(y)

        it = np.nditer(self.params, flags=['multi_index'])
        for _ in it:
            idx = it.multi_index
            params_plus = self.params.copy()
            params_plus[idx] += shift
            params_minus = self.params.copy()
            params_minus[idx] -= shift

            scores_plus = np.array([self.forward(x, params_plus) for x in X])
            scores_minus = np.array([self.forward(x, params_minus) for x in X])
            probs_plus = (scores_plus + 1) / 2
            probs_minus = (scores_minus + 1) / 2

            dprob_dtheta = (probs_plus - probs_minus) / 2.0
            grad[idx] = np.sum(dL_dprob * dprob_dtheta)
        return grad

    def fit(self, X, y, epochs=40, lr=0.35, verbose=None, epoch_callback=None):
        """Train variational parameters with parameter-shift gradients."""
        history = []
        for epoch in range(epochs):
            grad = self.parameter_shift_gradients(X, y)
            self.params -= lr * grad
            loss = self.compute_loss(X, y)
            history.append(loss)
            if verbose and (epoch % verbose == 0 or epoch == epochs - 1):
                print(f"  [qiskit-{self.backend_mode}] epoch {epoch+1:3d}/{epochs}  loss={loss:.4f}")
            if epoch_callback:
                epoch_callback(epoch + 1, epochs, loss)
        return history
