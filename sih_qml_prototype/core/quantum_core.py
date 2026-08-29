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

    def forward_batch(self, X, params=None):
        """Run the circuit in a vectorized batch for inputs X of shape (B, n_qubits).
        Returns an array of scores in [-1, 1] (average <Z> over all qubits per sample)."""
        if params is None:
            params = self.params
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        B = len(X)
        n_q = self.n_qubits

        # Initialize state tensor: shape (B, 2, 2, 2, 2)
        state = np.zeros((B,) + (2,) * n_q, dtype=complex)
        state[:, 0, 0, 0, 0] = 1.0

        # Encoding: RY(pi * X[:, q])
        for q in range(n_q):
            thetas = np.pi * X[:, q]
            c = np.cos(thetas / 2.0)
            s = np.sin(thetas / 2.0)
            c_exp = c.reshape((B,) + (1,) * (n_q - 1))
            s_exp = s.reshape((B,) + (1,) * (n_q - 1))

            idx0 = [slice(None)] * (n_q + 1)
            idx0[q + 1] = 0
            idx1 = [slice(None)] * (n_q + 1)
            idx1[q + 1] = 1

            s0 = state[tuple(idx0)].copy()
            s1 = state[tuple(idx1)].copy()
            state[tuple(idx0)] = c_exp * s0 - s_exp * s1
            state[tuple(idx1)] = s_exp * s0 + c_exp * s1

        # Variational layers
        for layer in range(self.n_layers):
            for q in range(n_q):
                theta_y, theta_z = params[layer, q]
                # RY(theta_y)
                cy, sy = np.cos(theta_y / 2.0), np.sin(theta_y / 2.0)
                idx0 = [slice(None)] * (n_q + 1)
                idx0[q + 1] = 0
                idx1 = [slice(None)] * (n_q + 1)
                idx1[q + 1] = 1

                s0 = state[tuple(idx0)].copy()
                s1 = state[tuple(idx1)].copy()
                state[tuple(idx0)] = cy * s0 - sy * s1
                state[tuple(idx1)] = sy * s0 + cy * s1

                # RZ(theta_z)
                ez0 = np.exp(-1j * theta_z / 2.0)
                ez1 = np.exp(1j * theta_z / 2.0)
                state[tuple(idx0)] *= ez0
                state[tuple(idx1)] *= ez1

            # Ring entanglement CNOT: q -> (q+1)%n_q
            for q in range(n_q):
                target = (q + 1) % n_q
                idx_c1 = [slice(None)] * (n_q + 1)
                idx_c1[q + 1] = 1
                idx_t0 = list(idx_c1)
                idx_t0[target + 1] = 0
                idx_t1 = list(idx_c1)
                idx_t1[target + 1] = 1

                temp = state[tuple(idx_t0)].copy()
                state[tuple(idx_t0)] = state[tuple(idx_t1)]
                state[tuple(idx_t1)] = temp

        # Average Pauli-Z expectation <Z> across qubits per sample
        z_sum = np.zeros(B)
        reduce_axes = tuple(range(1, n_q))
        for q in range(n_q):
            idx0 = [slice(None)] * (n_q + 1)
            idx0[q + 1] = 0
            idx1 = [slice(None)] * (n_q + 1)
            idx1[q + 1] = 1
            p0 = np.sum(np.abs(state[tuple(idx0)]) ** 2, axis=reduce_axes)
            p1 = np.sum(np.abs(state[tuple(idx1)]) ** 2, axis=reduce_axes)
            z_sum += (p0 - p1)

        scores = z_sum / n_q
        return scores

    def forward(self, x, params=None):
        """Run the circuit for one input sample x (length n_qubits, values in [-1,1]).
        Returns a score in [-1, 1] (average <Z> over all qubits)."""
        x_arr = np.asarray(x)
        if x_arr.ndim == 1:
            return float(self.forward_batch(x_arr.reshape(1, -1), params)[0])
        return self.forward_batch(x_arr, params)

    def predict_proba(self, X, params=None):
        """Map circuit output in [-1,1] to a probability in [0,1]."""
        scores = self.forward_batch(X, params)
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
        shift = np.pi / 2.0
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

            scores_plus = self.forward_batch(X, params_plus)
            scores_minus = self.forward_batch(X, params_minus)
            probs_plus = (scores_plus + 1.0) / 2.0
            probs_minus = (scores_minus + 1.0) / 2.0

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
