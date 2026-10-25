# Windows 10/11 & Visual Studio Code Environment Setup Guide

## Zero-Day GraphIDS

This guide details the complete, verified process for launching Zero-Day GraphIDS from scratch on Windows using PowerShell and Visual Studio Code without requiring CUDA or Docker.

---

## 1. Prerequisites Installation & Verification

Ensure Python 3.10, 3.11, 3.12, or 3.13 is installed on your Windows machine.

Open PowerShell as Administrator and verify:
```powershell
python --version
git --version
```

If Python is not recognized, add it to your Windows System PATH (`sysdm.cpl` > Advanced > Environment Variables).

---

## 2. Opening the Project in VS Code

```powershell
# Navigate to the workspace directory
cd C:\522\0day\Zero-Day-GraphIDS

# Launch in Visual Studio Code
code .
```

---

## 3. Creating & Activating Virtual Environment

Execute the following commands in the VS Code integrated PowerShell terminal:

```powershell
# Create virtual environment (with access to system site packages for fast torch setup)
python -m venv venv --system-site-packages

# If PowerShell script execution policy blocks activation, permit execution for current session:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# Activate virtual environment
.\venv\Scripts\Activate.ps1
```

*(You will observe `(venv)` prefixed on your PowerShell prompt).*

---

## 4. Upgrading Pip & Installing Dependencies

```powershell
# Upgrade pip to latest release
python -m pip install --upgrade pip

# Install complete requirements
pip install -r requirements.txt
```

---

## 5. Environment Variable Configuration

```powershell
# Copy the example environment file to active .env
Copy-Item .env.example .env
```

Verify that `.env` contains:
```ini
SECRET_KEY=django-insecure-zero-day-graphids-local-dev-key
DEBUG=True
ALLOWED_HOSTS=127.0.0.1,localhost
USE_POSTGRES=False
ML_DEVICE=cpu
```

---

## 6. Running Environment Diagnostics

```powershell
python scripts/check_environment.py
```
*Expected Output:*
```text
[+] Python version check: PASSED
[+] PyTorch: PASSED (Version: 2.8.0+cpu)
[+] PyTorch Geometric: PASSED
[+] Django Framework: PASSED
[+] All directories VERIFIED
```

---

## 7. Database Migrations & Initial Setup

```powershell
# Create Django migrations
python manage.py makemigrations accounts datasets ml_engine training detection dashboard reports

# Apply migrations to local SQLite database
python manage.py migrate
```

---

## 8. Creating SOC Administrator Account

```powershell
python manage.py createsuperuser
```
*(Enter username e.g. `admin`, email `admin@soc.local`, and password `AdminPassword123!`)*.

---

## 9. Generating Synthetic Network Flow Benchmark Datasets

```powershell
python scripts/create_sample_data.py
```
*Expected Output:*
```text
[+] Created: sample_data/CICIDS2017_sample.csv (1000 flows)
[+] Created: sample_data/CTU13_sample.csv (800 flows)
[+] Created: sample_data/USTC_TFC2016_sample.csv (800 flows)
[+] Created: sample_data/uninspected_traffic.csv (500 flows)
```

---

## 10. Running the Complete Demonstration Pipeline

```powershell
python scripts/run_pipeline.py
```
This executes the self-supervised GNN training loop, calculates normal baseline representations, calibrates the detection threshold on validation flows, and evaluates zero-day detection on uninspected network traffic.

---

## 11. Launching the Django Development Web Server

```powershell
python manage.py runserver 127.0.0.1:8000
```

Open your web browser and navigate to:
**`http://127.0.0.1:8000/`**

1. Log in with your admin credentials.
2. Explore the SOC Command Center Dashboard.
3. Ingest `sample_data/CICIDS2017_sample.csv` in **Datasets**.
4. Launch a self-supervised training job in **GNN Training**.
5. Upload `sample_data/uninspected_traffic.csv` in **Scan Traffic**.
6. Review chunk classifications, alerts, and download forensic PDF reports.

---

## 12. Running Automated Tests

```powershell
# Run the full test suite
python manage.py test

# Or run via pytest
pytest
```

---

## 13. Common Troubleshooting

| Issue / Error | Root Cause | Solution |
|---|---|---|
| `PSSecurityException` during `Activate.ps1` | Windows Restricted Execution Policy | Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` before activating venv. |
| `ImportError: DLL load failed while importing _sqlite3` | Corrupt local Python installation | Reinstall standard 64-bit Python release from python.org with default tcl/tk and sqlite options. |
| `OSError: [WinError 126] The specified module could not be found (torch)` | Missing Visual C++ Redistributable | Install the latest Microsoft Visual C++ 2015-2022 Redistributable (x64). |
| `Page Not Found (404) on /media/` | Debug media routing inactive | Ensure `DEBUG=True` in `.env`. Django serves media automatically in debug mode. |
