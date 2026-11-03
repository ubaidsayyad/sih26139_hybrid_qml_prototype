# Hybrid Quantum ML Platform for Early Disease Detection — Prototype
### SIH26139 · Egreen Quanta · MedTech/BioTech/HealthTech

A working, end-to-end prototype of the platform described in the problem statement:
data ingestion → classical pre-processing → **hybrid quantum-classical model** →
prediction → explainability → benchmarking against classical baselines — wrapped
in an interactive web dashboard.

## Why this is a genuine "hybrid quantum" system, not a simulation-only gimmick

`core/quantum_core.py` is a **from-scratch numpy statevector quantum simulator**:
real gate matrices (RY, RZ), real CNOT entanglement via index manipulation on the
2ⁿ-dimensional statevector, real Pauli-Z expectation-value measurement, and training
via the **analytic parameter-shift rule** — the actual gradient rule used on physical
quantum hardware (not a classical backprop approximation).

It was built this way instead of using Qiskit/PennyLane only because this dev
environment has no internet access to install them. The circuit logic is standard
NISQ hardware-efficient-ansatz VQC math, so swapping in a Qiskit/PennyLane/AWS
Braket backend (or real hardware) later is a contained change to one file —
`quantum_core.py` — the rest of the pipeline is untouched.

## Architecture

```
Biomedical data → Classical scaling + PCA (30 → 4 features)
                → Angle encoding (RY(π·x) per qubit, 4 qubits)
                → Variational layers ×2 (RY, RZ per qubit + ring CNOT entanglement)
                → Measurement: average ⟨Z⟩ across qubits
                → Sigmoid-mapped prediction
                → Permutation-importance explainability
                → Benchmarked vs Logistic Regression & Random Forest
```

## What's implemented

| Component | File | Notes |
|---|---|---|
| Data ingestion + preprocessing | `core/data_pipeline.py` | Breast Cancer Wisconsin dataset (swap in any tabular biomedical dataset) |
| Classical baselines | `core/classical_baseline.py` | Logistic Regression, Random Forest |
| Quantum circuit simulator | `core/quantum_core.py` | Hand-rolled statevector sim, parameter-shift training |
| Explainability | `core/explainability.py` | Model-agnostic permutation importance |
| Charts | `core/charts.py` | Comparison, ROC, training curve, importance bar charts |
| Orchestration | `core/pipeline_runner.py`, `run_pipeline.py` | CLI + web-callable pipeline |
| Web dashboard | `app.py`, `templates/`, `static/` | Flask app — run, visualize, live-predict |

## Running it

```bash
pip install flask numpy scipy scikit-learn pandas matplotlib
python3 app.py
# open http://localhost:5000
```

Or run the CLI version (prints a full report to the terminal and saves
`outputs/results.json`):

```bash
python3 run_pipeline.py
```

### What you'll see
1. **Overview page** — architecture diagram, "Run pipeline" button (adjustable epochs).
2. **Dashboard** — Accuracy/Precision/Recall/F1/AUC/train-time table for all four
   models (LogReg-30-feat, RandomForest-30-feat, LogReg-on-same-4-PCA-feat, Hybrid VQC),
   ROC curve, training loss curve, and side-by-side permutation-importance charts.
3. **Try it** — pick any held-out test patient and get a live quantum-model prediction
   with confidence and correctness.

### A real, measured result from this prototype (4 qubits, 2 layers, 45 epochs)

| Model | Accuracy | F1 | AUC | Train time |
|---|---|---|---|---|
| Logistic Regression (30 features) | 96.5% | 0.951 | 0.996 | 0.005s |
| Random Forest (30 features) | 95.8% | 0.940 | 0.994 | 0.353s |
| Logistic Regression (same 4 PCA features as quantum) | 95.1% | 0.929 | 0.998 | 0.002s |
| **Hybrid VQC (4 qubits)** | **88.8%** | **0.837** | **0.964** | 114.5s |

**Read this honestly in your pitch**: on a classical simulator with a small ansatz,
the quantum model currently trails the classical baseline even on identical features —
that's expected and documented in the QML literature for small-scale NISQ simulators.
The value to demonstrate to judges is the **working, fully-benchmarked pipeline**
(encoding → training → evaluation → explainability) and the honest, transparent
comparison methodology — not an inflated "quantum beats classical" claim. Judges will
respect the rigor of the "same-4-feature" fair-comparison row far more than an
overclaim.

## Suggested next steps before the finals

1. **Swap in Qiskit or PennyLane** once you have internet access, to run on a real
   quantum backend (IBM Quantum, AWS Braket) for at least one slide/demo — keep this
   simulator as the offline fallback.
2. **Add a second disease dataset** (e.g. UCI Heart Disease or PIMA Diabetes) to show
   the platform generalizes beyond one condition.
3. **Add a classical deep-learning arm for imaging data** (e.g. a small CNN on
   histopathology patches) if you want to demonstrate the "multi-modal" angle from
   the problem statement (genomics/imaging/EHR).
4. **Add SHAP** for the classical baseline (pip install shap) as an additional,
   more standard explainability method alongside permutation importance.
5. **Try more qubits/layers and compare wall-clock cost** — this is a good slide:
   show how training time scales, which motivates the "hybrid, not fully quantum"
   design choice for near-term hardware.
6. **Add authentication + patient data upload (CSV)** if you want a more
   product-like demo rather than a fixed dataset.

## File map

```
sih_qml_prototype/
├── app.py                     Flask web app
├── run_pipeline.py            CLI entrypoint (prints full report)
├── core/
│   ├── quantum_core.py        Statevector simulator + VQC + parameter-shift training
│   ├── data_pipeline.py       Ingestion + preprocessing (scaling, PCA)
│   ├── classical_baseline.py  LogReg / RandomForest baselines
│   ├── explainability.py      Permutation importance
│   ├── charts.py              Matplotlib chart generation
│   └── pipeline_runner.py     Orchestrates the full run for the web app
├── templates/                 Flask/Jinja2 HTML (index, dashboard, predict)
├── static/                    CSS + generated chart PNGs
└── outputs/                   results.json from CLI runs
```
