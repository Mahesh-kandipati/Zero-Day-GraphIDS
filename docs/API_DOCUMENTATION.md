# API & Endpoint Documentation

## Zero-Day GraphIDS

---

## 1. Authentication Endpoints (`accounts/`)

| Method | Endpoint | Description | Access |
|---|---|---|---|
| `GET/POST` | `/accounts/login/` | Authenticates SOC operator credentials. | Public |
| `GET/POST` | `/accounts/register/` | Provisions new security analyst account. | Public |
| `GET` | `/accounts/logout/` | Terminates active session. | Authenticated |
| `GET/POST` | `/accounts/profile/` | Inspects and updates operator profile. | Authenticated |

---

## 2. Dataset Management Endpoints (`datasets/`)

| Method | Endpoint | Description | Access |
|---|---|---|---|
| `GET` | `/datasets/` | Lists uploaded network flow benchmark datasets. | Authenticated |
| `GET/POST` | `/datasets/upload/` | Ingests new CSV, sanitizes columns, creates default graph configs. | Authenticated |
| `GET/POST` | `/datasets/<pk>/` | Inspects dataset statistics, column balance, and configures graph parameters. | Authenticated |
| `POST` | `/datasets/<pk>/delete/` | Removes dataset and associated disk files. | Authenticated |

---

## 3. GNN Training & Optimization Endpoints (`training/`)

| Method | Endpoint | Description | Access |
|---|---|---|---|
| `GET` | `/training/` | Displays training run history, loss trajectories, and F1 scores. | Authenticated |
| `GET/POST` | `/training/new/` | Dispatches new self-supervised GNN training run. | Authenticated |
| `GET` | `/training/<pk>/` | Real-time loss curves, epoch breakdown, and checkpoint weights. | Authenticated |
| `POST` | `/training/<pk>/retrain/`| Clones and restarts a training job configuration. | Authenticated |

---

## 4. Anomaly Detection & Incident Endpoints (`detection/`)

| Method | Endpoint | Description | Access |
|---|---|---|---|
| `GET/POST` | `/detection/upload/` | Ingests uninspected traffic CSV and runs graph inference. | Authenticated |
| `GET` | `/detection/result/<pk>/`| Displays chunk-level scores, threshold comparisons, and verdicts. | Authenticated |
| `GET` | `/detection/history/` | Archive of past traffic scans. | Authenticated |
| `GET` | `/detection/alerts/` | SOC alert triage board. | Authenticated |
| `POST` | `/detection/alerts/<pk>/update/` | Updates alert state (`NEW`, `IN_REVIEW`, `RESOLVED`, `FALSE_POSITIVE`). | Authenticated |

---

## 5. Forensic Reports & Data Export Endpoints (`reports/`)

| Method | Endpoint | Description | Access |
|---|---|---|---|
| `GET` | `/reports/` | Archive of all generated forensic reports. | Authenticated |
| `GET` | `/reports/generate/training/<job_id>/` | Compiles PDF audit report for GNN training run. | Authenticated |
| `GET` | `/reports/generate/detection/<upload_id>/` | Compiles PDF forensic report for traffic scan. | Authenticated |
| `GET` | `/reports/download/<pk>/` | Securely streams downloadable PDF report file. | Authenticated |
| `GET` | `/reports/export/csv/<upload_id>/` | Exports chunk anomaly score table as CSV. | Authenticated |
| `GET` | `/reports/export/json/<upload_id>/` | Exports full scan telemetry as structured JSON. | Authenticated |
