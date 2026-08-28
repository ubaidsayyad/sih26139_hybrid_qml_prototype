"""
quantum_core.py
----------------
A minimal, dependency-free (numpy-only) statevector quantum simulator used to
build the Variational Quantum Classifier (VQC) for the hybrid QML platform.

Why hand-rolled instead of Qiskit / PennyLane?
This environment has no internet access to install those packages, so the
circuit math (state preparation, parameterised rotations, entanglement,
expectation values, parameter-shift gradients) is implemented directly with
numpy. Swapping this module for a Qiskit/PennyLane backend later is a drop-in
change -- the rest of the pipeline (data_pipeline.py, classical_baseline.py,
evaluate.py) does not need to change.

Circuit design (hardware-efficient ansatz, NISQ-friendly):

    |0>--RY(pi*x0)--RY(theta)--RZ(theta)--o-----------X--
    |0>--RY(pi*x1)--RY(theta)--RZ(theta)--X--o--------|--
    |0>--RY(pi*x2)--RY(theta)--RZ(theta)-----X--o-----|--
    |0>--RY(pi*x3)--RY(theta)--RZ(theta)--------X--o--o--
         [ encoding ]  [   variational layer x L   ]

    Measurement: expectation value <Z> on every qubit, averaged.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Single- and two-qubit gate primitives
# ---------------------------------------------------------------------------

def _ry(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _rz(theta):
    return np.array([[np.exp(-1j * theta / 2), 0],
                      [0, np.exp(1j * theta / 2)]], dtype=complex)


_I2 = np.eye(2, dtype=complex)


def _apply_single_qubit_gate(state, gate, qubit, n_qubits):
    """Apply a 2x2 gate to `qubit` of an n_qubits statevector via tensor reshape."""
    state = state.reshape([2] * n_qubits)
    state = np.tensordot(gate, state, axes=([1], [qubit]))
    state = np.moveaxis(state, 0, qubit)
    return state.reshape(-1)


def _apply_cnot(state, control, target, n_qubits):
    """Apply CNOT(control -> target) via index manipulation (no dense matrix)."""
    state = state.reshape([2] * n_qubits)
    idx = [slice(None)] * n_qubits
    idx[control] = 1
    sub = state[tuple(idx)]
    state[tuple(idx)] = np.take(sub, [1, 0], axis=target if target < control else target - 1)
    return state.reshape(-1)


def _expectation_z(state, qubit, n_qubits):
    """<Z> on a given qubit: probability(0) - probability(1)."""
    state = state.reshape([2] * n_qubits)
    idx0 = [slice(None)] * n_qubits
    idx0[qubit] = 0
    idx1 = [slice(None)] * n_qubits
    idx1[qubit] = 1
    p0 = np.sum(np.abs(state[tuple(idx0)]) ** 2)
    p1 = np.sum(np.abs(state[tuple(idx1)]) ** 2)
    return float(p0 - p1)


# ---------------------------------------------------------------------------
# Variational Quantum Circuit
# ---------------------------------------------------------------------------

class VariationalQuantumCircuit:
    """
    A hardware-efficient ansatz VQC.

    n_qubits : number of qubits == number of input features (angle encoding)
    n_layers : number of variational (RY, RZ + ring-CNOT) layers
    """

    def __init__(self, n_qubits, n_layers, seed=42):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        rng = np.random.default_rng(seed)
        # params shape: (n_layers, n_qubits, 2) -> RY, RZ angle per qubit per layer
        self.params = rng.uniform(-0.1, 0.1, size=(n_layers, n_qubits, 2))

    def n_params(self):
        return self.params.size

    def _initial_state(self):
        state = np.zeros(2 ** self.n_qubits, dtype=complex)
        state[0] = 1.0
        return state

    def _encode(self, state, x):
        """Angle encoding: RY(pi * x_i) on each qubit."""
        for q in range(self.n_qubits):
            state = _apply_single_qubit_gate(state, _ry(np.pi * x[q]), q, self.n_qubits)
        return state

    def _variational_layer(self, state, layer_params):
        for q in range(self.n_qubits):
            theta_y, theta_z = layer_params[q]
            state = _apply_single_qubit_gate(state, _ry(theta_y), q, self.n_qubits)
            state = _apply_single_qubit_gate(state, _rz(theta_z), q, self.n_qubits)
        # ring entanglement
        for q in range(self.n_qubits):
            state = _apply_cnot(state, q, (q + 1) % self.n_qubits, self.n_qubits)
        return state

    def forward(self, x, params=None):
        """Run the circuit for one input sample x (length n_qubits, values in [-1,1]).
        Returns a score in [-1, 1] (average <Z> over all qubits)."""
        if params is None:
            params = self.params
        state = self._initial_state()
        state = self._encode(state, x)
        for layer in range(self.n_layers):
            state = self._variational_layer(state, params[layer])
        z_vals = [_expectation_z(state, q, self.n_qubits) for q in range(self.n_qubits)]
        return float(np.mean(z_vals))

    def predict_proba(self, X, params=None):
        """Map circuit output in [-1,1] to a probability in [0,1]."""
        scores = np.array([self.forward(x, params) for x in X])
        return (scores + 1.0) / 2.0

    # -----------------------------------------------------------------
    # Parameter-shift gradient (true quantum gradient rule, not autodiff)
    # -----------------------------------------------------------------
    def _binary_cross_entropy_grad_wrt_scores(self, probs, y):
        eps = 1e-7
        probs = np.clip(probs, eps, 1 - eps)
        return probs - y  # d(BCE)/d(sigmoid input), reused as chain-rule proxy

    def compute_loss(self, X, y, params=None):
        probs = self.predict_proba(X, params)
        eps = 1e-7
        probs = np.clip(probs, eps, 1 - eps)
        return float(-np.mean(y * np.log(probs) + (1 - y) * np.log(1 - probs)))

    def parameter_shift_gradients(self, X, y):
        """Compute the gradient of the BCE loss w.r.t every circuit parameter
        using the analytic parameter-shift rule: dE/dtheta = [E(theta+pi/2) - E(theta-pi/2)] / 2
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
        """Train with plain gradient descent + parameter-shift gradients."""
        history = []
        for epoch in range(epochs):
            grad = self.parameter_shift_gradients(X, y)
            self.params -= lr * grad
            loss = self.compute_loss(X, y)
            history.append(loss)
            if verbose and (epoch % verbose == 0 or epoch == epochs - 1):
                print(f"  [quantum] epoch {epoch+1:3d}/{epochs}  loss={loss:.4f}")
            if epoch_callback:
                epoch_callback(epoch + 1, epochs, loss)
        return history
