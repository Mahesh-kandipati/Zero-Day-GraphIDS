# Automated Testing & Verification Guide

## Zero-Day GraphIDS

---

## 1. Overview of Test Coverage

The automated test suite verifies all four tiers of the application:
1. **Accounts & RBAC**: CustomUser models, login flows, role permissions, profile protection.
2. **Datasets & Preprocessing**: File validation, column sanitation, infinite/null imputation, leakage-free train/val/test splitting.
3. **ML Engine & GNN**: $k$-NN graph builder, GraphSAGE forward pass, self-supervised InfoNCE/DGI loss, backpropagation, and safe checkpoint loading.
4. **Detection & Evaluation**: Mahalanobis distance scoring, adaptive threshold calibration, leave-one-attack-family-out metrics, and baseline comparisons against Isolation Forest.
5. **Dashboard & Forensic Reports**: SOC telemetry views, Cytoscape graph serialization, ReportLab PDF generation.

---

## 2. Executing Automated Tests

### Run Full Test Suite via Django Test Runner
```powershell
python manage.py test
```

### Run Full Test Suite via Pytest
```powershell
pytest
```

### Run Individual Module Tests
```powershell
# Accounts module
python manage.py test accounts

# Datasets preprocessing & validation
python manage.py test datasets

# GNN and self-supervised loss
python manage.py test ml_engine

# Anomaly scoring and thresholding
python manage.py test detection

# Forensic evaluation framework
python manage.py test training

# Forensic report compilation
python manage.py test reports
```

---

## 3. End-to-End Test Scenario

To verify the complete integrated system from data ingestion to zero-day anomaly verdict:
```powershell
python scripts/run_pipeline.py
```
This script exercises:
- Environment diagnostic check.
- Synthetic network flow dataset synthesis.
- GNN self-supervised optimization over normal flows.
- Reference profile calculation and threshold calibration.
- Zero-day inference on uninspected traffic.
- Benchmark comparison against Isolation Forest.
