# Zero-Day GraphIDS

> **A Self-Supervised Graph Learning Framework for Intelligent Zero-Day Malicious Traffic Detection**

Zero-Day GraphIDS is a production-ready, full-stack cybersecurity application designed to detect novel, previously unseen zero-day network attacks without relying on labeled malicious training data. By converting network flow streams into $k$-Nearest Neighbors ($k$-NN) similarity graphs and training GraphSAGE encoders with self-supervised reconstruction and contrastive objectives, the system establishes a robust topological baseline of normal operations and scores deviations using regularized Mahalanobis distance.

---

## 🌟 Key Features

- **Self-Supervised Graph Learning**: Eliminates reliance on attack signatures or supervised labels; trains exclusively on baseline benign network flows.
- **Topological & Statistical Representations**: Builds $k$-NN metric proximity graphs over temporal flow chunks, capturing local relational structure.
- **Deep Graph Infomax & Feature Reconstruction**: Jointly optimizes local-to-global mutual information and node feature reconstruction.
- **Robust Zero-Day Anomaly Detection**: Employs Ledoit-Wolf regularized Mahalanobis distance with adaptive threshold calibration strictly on validation sets.
- **Leave-One-Attack-Family-Out Evaluation Protocol**: Empirically validates detection efficacy on completely withheld attack vectors and compares against Isolation Forest.
- **Professional SOC Operations Center**: Responsive dark cybersecurity dashboard built with Bootstrap 5, Chart.js, and Cytoscape.js.
- **Automated Forensic PDF Generation**: Multi-page PDF audit reports generated via ReportLab.
- **CPU & Windows Optimized**: 100% CPU compatible with optional CUDA GPU acceleration.

---

## 🏗️ System Architecture

```text
Raw Network Flow CSV
        │
        ▼
[ Preprocessing & Leakage Guard ]
        │
        ▼
[ k-NN Similarity Graph Construction ]
        │
        ▼
[ GraphSAGE Self-Supervised Encoder ]  <--- (Trained strictly on normal flows)
        │
        ▼
[ Latent Representation Embeddings h_G ]
        │
        ▼
[ Regularized Mahalanobis Distance Scoring ]
        │
        ▼
[ Adaptive Validation Threshold Calibration ]
        │
        ▼
[ Incident Verdict & Alerts ] ──> [ SOC Dashboard, Reports & Forensic PDF ]
```

---

## 🚀 Quick Start on Windows 10/11 (VS Code & PowerShell)

### 1. Prerequisites
Verify Python 3.10+ is installed:
```powershell
python --version
```

### 2. Virtual Environment Setup
```powershell
cd C:\522\0day\Zero-Day-GraphIDS
python -m venv venv --system-site-packages
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment
```powershell
Copy-Item .env.example .env
```

### 5. Run Environment Diagnostics
```powershell
python scripts/check_environment.py
```

### 6. Database Migrations & Admin Account
```powershell
python manage.py makemigrations accounts datasets ml_engine training detection dashboard reports
python manage.py migrate
python manage.py createsuperuser
```

### 7. Generate Synthetic Network Flow Benchmarks
```powershell
python scripts/create_sample_data.py
```

### 8. Run Complete End-to-End Demonstration Pipeline
```powershell
python scripts/run_pipeline.py
```

### 9. Launch the Django Web Server
```powershell
python manage.py runserver 127.0.0.1:8000
```
Open **`http://127.0.0.1:8000/`** in your browser.

---

## 🧪 Automated Testing

Execute the comprehensive unit and integration test suite:
```powershell
# Run via Django Test Runner
python manage.py test

# Or run via pytest
pytest
```

---

## 👥 Four-Member Engineering Division

- **Member 1 (Full-Stack & SOC Lead)**: Django core, RBAC, accounts, dashboard visualizations, responsive dark theme.
- **Member 2 (Data Pipeline Lead)**: Dataset ingestion, file validation, schema detection, preprocessing, and data leakage prevention.
- **Member 3 (GNN Specialist)**: Graph construction, GraphSAGE encoder, self-supervised InfoNCE/DGI, feature reconstruction.
- **Member 4 (Cybersecurity & Evaluation Specialist)**: Mahalanobis anomaly scoring, threshold calibration, zero-day evaluation, baselines, and ReportLab PDF forensic engine.

---

## 📄 License & Disclaimer

This software is developed strictly for defensive cybersecurity research, authorized intrusion detection, and educational evaluation.
