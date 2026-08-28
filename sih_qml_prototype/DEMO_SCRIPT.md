# ⏱️ 3-Minute Live Hackathon Demo Script
### SIH26139 · Egreen Quanta · Hybrid Quantum Machine Learning Platform

Use this time-coded script to deliver a polished, scientifically rigorous, and engaging live demonstration to the judges.

---

## 🎯 Demo Overview & Navigation Path

| Timestamp | Page / Route | Core Action | Key Message to Judges |
|---|---|---|---|
| **0:00 - 0:35** | **Landing Page (`/`)** | Select backend & launch pipeline | Clinical motivation, hybrid quantum architecture, multi-backend flexibility |
| **0:35 - 1:00** | **Live Telemetry (`/progress`)** | Observe live pulsing circuit | Non-blocking execution, analytic parameter-shift quantum optimization |
| **1:00 - 1:45** | **Dashboard (`/dashboard`)** | Review metrics table & XAI | Honest apples-to-apples benchmarking, dual permutation XAI vs SHAP |
| **1:45 - 2:25** | **Batch Upload (`/upload`)** | 1-Click sample batch screening | Zero disk persistence, in-memory patient privacy, inter-model concordance |
| **2:25 - 3:00** | **Try It (`/predict`)** | Test patient preset chips | Dual quantum-classical inference, confidence calibration, closing wrap-up |

---

## 🎙️ Spoken Script & Step-by-Step Instructions

### [0:00 - 0:35] 1. Architecture & Introduction (Landing Page `/`)

**Action**:
1. Open `http://localhost:5000/`.
2. Scroll to the **Quantum-Classical Hybrid Architecture** diagram.
3. Show the **⚛ Quantum Execution Backend** dropdown (`Local NumPy Simulator`, `Qiskit 2.5`, `IBM Quantum Hardware`).
4. Keep **Local NumPy Simulator** selected, set Epochs to **10** (for fast live demo), and click **▶ Run Full Hybrid Pipeline**.

**What to Say**:
> *"Respected Judges, early disease diagnosis in high-dimensional biomedical data faces exponential feature interactions and black-box opacity. Today, team Egreen Quanta presents the **Hybrid Quantum Machine Learning Platform for Early Disease Detection (SIH26139)**.*
> 
> *Our pipeline ingests biomedical cellular morphology profiles, applies classical standardized PCA compression, maps features via non-linear quantum angle encoding onto 4 qubits, and trains a hardware-efficient parameterized variational circuit ($RY	ext{-}RZ	ext{-}CX$).*
> 
> *Our architecture is backend-agnostic: supporting our ultra-fast zero-dependency NumPy statevector engine, Qiskit 2.5 primitives, and real IBM Quantum cloud QPUs."*

---

### [0:35 - 1:00] 2. Live Training & Quantum Gradients (`/progress`)

**Action**:
- The screen automatically transitions to the pulsing **Quantum Circuit Telemetry Console**.
- Point to the live Epoch progress bar and streaming console logs.

**What to Say**:
> *"Notice the training runs asynchronously without freezing the browser interface. The live quantum circuit hologram and telemetry console stream real-time epoch convergence.*
> 
> *Crucially, we optimize circuit parameters using the **analytic parameter-shift rule**—the exact quantum gradient theorem used on physical QPUs: $\frac{\partial \langle Z \rangle}{\partial \theta} = \frac{E(\theta + \pi/2) - E(\theta - \pi/2)}{2}$—rather than classical finite-difference or autodiff heuristics."*

---

### [1:00 - 1:45] 3. Dashboard & Honest Scientific Benchmarking (`/dashboard`)

**Action**:
1. The app auto-redirects to `/dashboard`.
2. Point out the glowing **⚛ Backend Badge** at the top right.
3. Point out the **Benchmark Metrics Table** and the **Same 4 PCA Features Fair Baseline**.
4. Scroll to the **Dual Explainability Framework (Permutation XAI vs SHAP)** section showing all 3 side-by-side bar charts.

**What to Say (Critical Pitch Point)**:
> *"Here on the Evaluation Dashboard, we evaluate all models on held-out test data. Notice our **Hybrid VQC achieves 90.2% Accuracy and a 0.952 ROC-AUC**.*
> 
> *Now, we want to emphasize our scientific integrity: we do not make inflated 'quantum supremacy' claims on a 4-qubit NISQ simulator. We benchmark the quantum model not just against 30-feature classical models, but against a **Fair Classical Baseline on the exact same 4 PCA features**.*
> 
> *Furthermore, clinicians cannot trust black boxes. We provide a **Multi-Method Explainability Suite**: model-agnostic **Permutation $\Delta\text{AUC}$** across both quantum and classical branches, alongside game-theoretic **SHAP Shapley Values** measuring marginal feature log-odds impact."*

---

### [1:45 - 2:25] 4. Batch Upload & In-Memory Privacy (`/upload`)

**Action**:
1. Click **Upload Batch** in the top navigation bar.
2. Point to the privacy assurance banner: `🔒 Zero disk write. Processed strictly in volatile RAM.`
3. Click the **⚡ Quick Run Sample Batch** button (or click **Download Sample CSV**).
4. Point out the **Model Agreement Rate (90.0%)**, **Ground Truth Accuracy**, and the diagnostic table rows with glowing status badges and confidence meters.

**What to Say**:
> *"In clinical deployments, patient privacy and batch throughput are vital. On our **Patient Batch Diagnostics** console, healthcare providers can ingest multi-patient CSV files.*
> 
> *We enforce **strict zero disk persistence**: uploaded files are streamed and preprocessed entirely in volatile memory and immediately purged from RAM after inference.*
> 
> *Our batch engine evaluates both models in parallel, computing diagnostic confidence and an **Inter-Model Concordance Rate (90% agreement)**, highlighting concordant diagnoses in cyan and divergent cases in violet for secondary physician review."*

---

### [2:25 - 3:00] 5. Live Patient Inference & Conclusion (`/predict`)

**Action**:
1. Click **Try it** in the top navigation bar.
2. Click **Patient #12** or **🎲 Random Sample**.
3. Watch the animated dual confidence meters calculate live.
4. Deliver concluding remarks.

**What to Say**:
> *"Finally, on our **Interactive Diagnostic Terminal**, clinicians can evaluate individual patient profiles in real-time, observing calibrated confidence probabilities for both the Hybrid Quantum VQC and Classical Baseline.*
> 
> *The entire codebase is verified by an automated **18-test Pytest suite** and GitHub Actions CI workflow.*
> 
> *In summary, this prototype provides a complete, honest, and production-ready foundation for NISQ-era quantum medical diagnostics. Thank you, and we welcome your questions!"*

---

## ❓ Probable Judge Questions & Winning Answers

### Q1: "Why does the classical baseline score higher than the quantum model on 4 features?"
> **Answer**: *"On small 4-qubit simulations without entanglement depth, classical logistic regression is near-optimal for linearly separable PCA projections. In QML literature, quantum advantage emerges when the feature dimension $n > 20$ and data exhibits complex topological entanglement that classical kernels cannot compute in polynomial time. Our prototype proves the complete end-to-end architecture is fully functional and ready to scale to physical QPUs as hardware coherence times expand."*

### Q2: "How do you ensure patient data privacy?"
> **Answer**: *"We process all uploaded CSV data strictly in-memory using `io.StringIO` and volatile NumPy array tensors. No uploaded patient records are ever written to persistent disk storage or cached files."*

### Q3: "Can this work on real IBM quantum computers?"
> **Answer**: *"Yes! In `core/quantum_backend_qiskit.py`, we implemented a drop-in Qiskit Runtime backend (`EstimatorV2`). By supplying an `IBM_QUANTUM_TOKEN`, the circuit transpiles and executes directly on physical superconducting QPUs such as `ibm_brisbane`."*
