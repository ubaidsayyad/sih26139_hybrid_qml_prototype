# 🚀 Windows Setup Guide — Hybrid Quantum ML Platform
### SIH26139 · Egreen Quanta · MedTech / Quantum HealthTech

A complete, beginner-friendly step-by-step guide to installing, configuring, running, and troubleshooting the **Hybrid Quantum Machine Learning Platform for Early Disease Detection** on Windows 10 & 11.

---

## 📋 System Prerequisites

- **Operating System**: Windows 10 or Windows 11 (64-bit)
- **Python**: Version **3.11** or **3.12** recommended (Python 3.10+ supported)
- **RAM**: Minimum 4 GB (8 GB+ recommended)
- **Disk Space**: ~500 MB free disk space

---

## 🛠️ Step 1: Install Python on Windows

1. Download the official installer from [python.org/downloads](https://www.python.org/downloads/).
2. Run the `.exe` installer.
3. ⚠️ **CRITICAL STEP**: On the first installer screen, **CHECK THE BOX** that says:
   > ☑ **Add python.exe to PATH**
4. Click **Install Now**.
5. Once installation finishes, click **Disable path length limit** (if prompted), then click **Close**.

### Verify Python in PowerShell:
Open **PowerShell** or **Command Prompt** and run:
```powershell
python --version
pip --version
```
*(You should see `Python 3.12.x` and `pip 24.x`)*

---

## 📂 Step 2: Open the Project Directory

Open **PowerShell** and navigate to the project directory:
```powershell
cd C:\Users\Admin\Downloads\sih26139_hybrid_qml_prototype\sih_qml_prototype
```

---

## 🐍 Step 3: Create & Activate a Virtual Environment (Recommended)

Creating an isolated virtual environment ensures clean dependency management:

```powershell
# Create a virtual environment named 'venv'
python -m venv venv

# Activate the virtual environment
.\venv\Scripts\Activate.ps1
```

*(You will see `(venv)` appear at the beginning of your terminal line)*

> **Note for Command Prompt (`cmd.exe`) users**:
> Run `venv\Scripts\activate.bat` instead.

---

## 📦 Step 4: Install Dependencies

Upgrade pip and install all required libraries:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

This installs:
- `flask`: Web application server and interactive REST API
- `numpy`, `scipy`, `pandas`: Vector math and matrix transformations
- `scikit-learn`: Classical ML baselines, PCA, and dataset benchmarks
- `matplotlib`: Dark-mode telemetry chart synthesis
- `qiskit`, `qiskit-ibm-runtime`: Real quantum circuit simulation and cloud QPU primitives
- `shap`: Game-theoretic Shapley explainability
- `pytest`: Automated test verification suite

---

## ⚛️ Step 5: Run the Application

Launch the Flask server:

```powershell
python app.py
```

You will see output similar to:
```
 * Serving Flask app 'app'
 * Debug mode: on
 * Running on http://127.0.0.1:5000 (Press CTRL+C to quit)
```

Now open your web browser and go to:
👉 **[http://localhost:5000](http://localhost:5000)** (or `http://127.0.0.1:5000`)

---

## 🧪 Step 6: Run the Automated Test Suite

To verify that every component (data pipeline, classical baselines, quantum circuits, explainability, batch upload, and Qiskit backend) is operating properly:

```powershell
python -m pytest -v tests/
```

Expected output: `18 passed in ~25s (100% test success rate)`.

---

## 🔑 Optional: Configure Real IBM Quantum Hardware

To execute on real physical superconducting QPUs via the IBM Quantum Cloud:

1. Create a free account at [quantum.ibm.com](https://quantum.ibm.com/).
2. Copy your 64-character API token from your dashboard.
3. Set the environment variable in your terminal before starting the app:
   ```powershell
   $env:IBM_QUANTUM_TOKEN="your_64_character_token_here"
   ```
   *(Or paste it directly into the web UI token field when launching the pipeline)*

---

## 🔧 Troubleshooting Common Windows Errors

### 1. PowerShell Script Execution Policy Error
**Error**: `cannot be loaded because running scripts is disabled on this system` when activating `venv`.  
**Solution**: Run this command in PowerShell to permit script execution for your user session:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 2. `'python'` is not recognized as an internal or external command
**Cause**: Python was installed without checking "Add python.exe to PATH".  
**Solution**:
1. Search Windows for "Environment Variables" and click **Edit the system environment variables**.
2. Click **Environment Variables** button.
3. Under **User variables**, select `Path` and click **Edit**.
4. Add `C:\Users\<YourUsername>\AppData\Local\Programs\Python\Python312` and `C:\Users\<YourUsername>\AppData\Local\Programs\Python\Python312\Scripts`.
5. Restart PowerShell.

### 3. Port 5000 Already in Use (`Address already in use` or `OSError: [WinError 10048]`)
**Cause**: Another instance of the Flask app or another service is listening on port 5000.  
**Solution**: Stop existing Python processes in PowerShell:
```powershell
Stop-Process -Name python -Force
```
Then start `python app.py` again.

### 4. Permission Error During `pip install` (`Access is denied` or `.exe.deleteme`)
**Cause**: Antivirus or file locks on Python scripts directory.  
**Solution**: Install with the `--user` flag:
```powershell
python -m pip install --user -r requirements.txt
```

### 5. Line Endings or Encoding Warnings on Windows Console (`UnicodeEncodeError: 'charmap'`)
**Solution**: Set the console encoding to UTF-8 in PowerShell:
```powershell
$env:PYTHONIOENCODING="utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
```
