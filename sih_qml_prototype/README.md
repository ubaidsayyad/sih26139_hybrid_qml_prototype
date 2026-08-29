# ⚛️ Hybrid Quantum Machine Learning Platform for Early Disease Detection
### Smart India Hackathon 2024 · SIH26139 · Egreen Quanta · MedTech / Quantum HealthTech

An end-to-end, scientifically rigorous hybrid quantum-classical machine learning platform for early disease detection, designed to bridge high-dimensional biomedical clinical data with near-term quantum advantage.

---

## 🌟 Executive Summary & Key Highlights

- **Hybrid Architecture**: Combines classical standardization and PCA feature extraction with a 4-qubit hardware-efficient variational quantum classifier ($RY	ext{-}RZ	ext{-}CX$ ring entanglement).
- **Analytic Parameter-Shift Gradients**: Optimizes circuit parameters using exact analytic quantum gradients rather than classical finite differences or autodiff approximations.
- **Multi-Backend Flexibility**:
  1. `simulator`: Ultra-low latency, dependency-free NumPy statevector engine.
  2. `qiskit_simulator`: Qiskit 2.5 Statevector simulation for circuit validation.
  3. `ibm_hardware`: Real superconducting quantum processor execution via IBM Quantum Runtime (`EstimatorV2`).
- **Asynchronous Execution & Live Telemetry**: Background threading with real-time epoch convergence streaming and an animated quantum circuit hologram.
- **Fair, Transparent Benchmarking**: Evaluated on held-out test data against full 30-feature classical models and a **Fair Classical Baseline on the exact same 4 PCA features**.
- **Dual Explainability Suite**: Model-agnostic **Permutation Feature Importance ($\Delta	ext{AUC}$)** across quantum and classical models alongside **Game-Theoretic SHAP (Shapley Values)**.
- **In-Memory Batch Diagnostics & Privacy**: Multi-patient CSV upload with strict schema validation and **zero disk persistence** to protect patient privacy.
- **Enterprise-Grade Quality**: **18/18 passing Pytest unit tests** and automated GitHub Actions CI workflow.

---

## 🏗️ Quantum-Classical Hybrid Architecture

```
Biomedical Data Ingestion (30D)
          │
          ▼
Classical Preprocessing (StandardScaler + PCA 30D ➔ 4D)
          │
          ▼
Quantum Angle Encoding (RY(π·x) on 4 Qubits)
          │
          ▼
Variational Layers ×2 (RY, RZ per qubit + Ring CNOT Entanglement)
          │
          ▼
Pauli-Z Observable Readout (Average ⟨Z⟩ across all qubits)
          │
          ▼
Sigmoid-Mapped Binary Probability P(Disease-Positive)
          │
    ┌─────┴─────────────────────────┐
    ▼                               ▼
Model-Agnostic XAI          Apples-to-Apples Benchmarking
(Permutation ΔAUC + SHAP)   (vs 30-Feat & 4-Feat Classical Baselines)
```

---

## ⚛️ Quantum Execution Backends

| Backend Key | Engine / Primitive | Target Use Case | Speed / Latency |
|---|---|---|---|
| **`simulator`** | `core/quantum_core.py` (NumPy tensor math) | Offline training, rapid local iteration, hackathon demos | **~0.6 ms** / pass (~12x faster) |
| **`qiskit_simulator`** | `core/quantum_backend_qiskit.py` (Qiskit 2.5 `Statevector`) | Open-source circuit compilation and gate validation | **~7.5 ms** / pass |
| **`ibm_hardware`** | `qiskit-ibm-runtime` (`EstimatorV2`) | Physical superconducting QPUs (`ibm_brisbane`, `ibm_kyoto`) | Cloud queue / QPU execution |

---

## 📊 Measured Benchmark Results (Held-Out Test Set)

| Model Architecture | Features / Parameters | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Train Time |
|---|---|---|---|---|---|---|---|
| **Logistic Regression (Full)** | 30 Features | 96.5% | 0.980 | 0.925 | 0.951 | 0.996 | 0.011s |
| **Random Forest (Full)** | 30 Features (200 trees) | 95.8% | 1.000 | 0.887 | 0.940 | 0.994 | 0.693s |
| **Logistic Regression (Fair)** | **Same 4 PCA Features** | **95.1%** | **1.000** | **0.868** | **0.929** | **0.998** | **0.008s** |
| **Hybrid VQC (Quantum)** | **4 Qubits (16 Parameters)** | **90.2%** | **0.868** | **0.868** | **0.868** | **0.952** | **62.8s** |

> **Honest Scientific Interpretation**: On 4-qubit classical simulation with low entanglement depth, linear models excel on linearly-separable PCA projections. The purpose of this prototype is establishing a complete, transparent, and audited quantum-classical pipeline that scales directly to physical multi-qubit QPUs as quantum coherence and feature dimensionality increase.

---

## 🔬 Multi-Method Explainability Framework

1. **Permutation Feature Importance ($\Delta	ext{AUC}$)**:
   - Evaluates the global decline in model discrimination under stochastic feature permutation.
   - Applied identically to both the **Hybrid Quantum VQC** and the **Classical Baseline**.
2. **SHAP (Shapley Additive exPlanations)**:
   - Computes game-theoretic marginal feature attributions ($\mathbb{E}[|\phi_i|]$) quantifying the exact impact of each morphological metric on shifting prediction log-odds.

---

## 🔒 In-Memory Batch Diagnostics & Privacy Assurance

- **Zero Disk Write**: Uploaded patient CSV files are parsed and evaluated strictly in volatile RAM (`io.StringIO` / memory tensors) and immediately purged after inference.
- **Strict Schema Validation**: Verifies all 30 standard cellular morphological features, numeric data integrity, and provides explicit error messages on invalid rows/columns.
- **Inter-Model Concordance**: Computes patient-level agreement (**Concordant** vs **Divergent**), enabling clinicians to prioritize discordant cases for secondary clinical review.
- **Downloadable Sample CSV**: `GET /download-sample` provides ready-to-test benchmark CSV records (`sample_patients_test_batch.csv`).

---

## 🚀 Quickstart Guide

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/your-org/sih26139_hybrid_qml_prototype.git
cd sih26139_hybrid_qml_prototype/sih_qml_prototype

# Install dependencies
python -m pip install -r requirements.txt
```

### 2. Running the Web Platform

#### Recommended for Live Demos & Production Presentations (Waitress WSGI):
```bash
python run_server.py
```
*Starts the multi-threaded Waitress WSGI production server on `http://localhost:5000` (8 worker threads, immune to file-write auto-reloader restarts, highly stable under background quantum training).*

#### For Local Code Development Only:
```bash
python app.py
```

Open **[http://localhost:5000](http://localhost:5000)** in your browser.

### 3. Running the Automated Test Suite
```bash
python -m pytest -v tests/
```
*(All 48 tests pass across data pipelines, classical baselines, quantum circuits, explainability, batch upload, authentication, PDF generation, REST API v1, and Qiskit backends)*

---

## 📖 Key Documentation Files

- **[`SETUP.md`](SETUP.md)**: Beginner-friendly Windows 10/11 step-by-step setup guide and comprehensive troubleshooting handbook.
- **[`DEMO_SCRIPT.md`](DEMO_SCRIPT.md)**: Time-coded 3-minute live presentation script for hackathon judges with key scientific talking points.
- **[`.env.example`](.env.example)**: Documented environment variable template for configuring IBM Quantum API tokens and runtime ports.

---

## 📂 Repository File Structure

```
sih_qml_prototype/
├── app.py                          # Flask web application & REST API
├── run_pipeline.py                 # CLI entrypoint for headless pipeline execution
├── requirements.txt                # Pinned production dependencies
├── SETUP.md                        # Windows beginner setup & troubleshooting guide
├── DEMO_SCRIPT.md                  # 3-minute live presentation script
├── .env.example                    # Environment variable configuration template
├── .github/
│   └── workflows/
│       └── tests.yml               # Automated GitHub Actions CI workflow
├── core/
│   ├── data_pipeline.py            # Biomedical data ingestion, scaling & PCA reduction
│   ├── quantum_core.py             # NumPy statevector simulator & parameter-shift VQC
│   ├── quantum_backend_qiskit.py   # Drop-in Qiskit 2.5 Statevector & IBM QPU backend
│   ├── classical_baseline.py       # Logistic Regression & Random Forest baselines
│   ├── explainability.py           # Permutation importance (ΔAUC) & SHAP explainability
│   ├── patient_batch.py            # In-memory CSV validation & dual-model batch screening
│   ├── charts.py                   # Matplotlib dark-mode telemetry visualization
│   └── pipeline_runner.py          # Orchestrates end-to-end multi-model pipelines
├── templates/
│   ├── base.html                   # Master layout with interactive quantum mesh background
│   ├── index.html                  # Landing page with backend selector & architecture flow
│   ├── progress.html               # Live asynchronous telemetry console & circuit hologram
│   ├── dashboard.html              # Results dashboard with 3-column XAI & metrics table
│   ├── predict.html                # Interactive single-patient diagnostic console
│   └── upload.html                 # Patient batch ingestion & in-memory diagnostic table
├── static/
│   ├── style.css                   # Cyber-futuristic glassmorphic stylesheet
│   └── *.png                       # Generated telemetry charts & screenshots
└── tests/
    ├── test_data_pipeline.py       # Tests for dataset ingestion & stratified splits
    ├── test_classical_baseline.py  # Tests for classical baseline evaluation
    ├── test_quantum_core.py        # Tests for quantum circuit bounds & gradients
    ├── test_explainability.py      # Tests for permutation importance & SHAP attributions
    ├── test_patient_batch.py       # Tests for CSV validation & zero persistence
    └── test_quantum_backend_qiskit.py # Tests for Qiskit-NumPy mathematical equivalence
```

---

## 👥 Authors & Acknowledgments

- **Team**: Egreen Quanta
- **Problem Statement**: SIH26139 (Smart India Hackathon 2024)
- **Theme**: MedTech / BioTech / Quantum HealthTech
