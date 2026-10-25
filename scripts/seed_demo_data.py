"""
Database Seeding Script for Zero-Day GraphIDS.
Creates initial admin account, ingests synthetic benchmark datasets,
trains a baseline GNN model, and runs a demo uninspected traffic scan.
"""

import os
import sys
import django
from django.core.files import File

# Initialize Django
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from accounts.models import CustomUser
from datasets.models import Dataset, GraphConfig, DatasetSplit
from training.models import TrainingJob
from training.trainer import run_self_supervised_training
from detection.models import TrafficUpload
from detection.detector import run_zero_day_detection

def seed():
    print("[*] Seeding database with initial SOC operator and demonstration telemetry...")

    # 1. Admin User
    admin_user, created = CustomUser.objects.get_or_create(
        username='admin',
        defaults={
            'email': 'admin@soc.local',
            'role': CustomUser.ROLE_ADMIN,
            'organization': 'Central Cyber Defense Command',
            'is_staff': True,
            'is_superuser': True
        }
    )
    if created:
        admin_user.set_password('AdminPassword123!')
        admin_user.save()
        print("[+] Provisioned Administrator: 'admin' (Password: 'AdminPassword123!')")
    else:
        print("[*] Administrator 'admin' already present.")

    # 2. Ingest Sample Dataset
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sample_csv = os.path.join(base_dir, 'sample_data', 'CICIDS2017_sample.csv')

    if os.path.exists(sample_csv) and not Dataset.objects.filter(name="CIC-IDS2017 Baseline Benchmark").exists():
        with open(sample_csv, 'rb') as f:
            dataset = Dataset.objects.create(
                name="CIC-IDS2017 Baseline Benchmark",
                description="Synthesized multi-vector benchmark with DDoS, PortScan, and ZERO_DAY_UNKNOWN traffic.",
                dataset_type=Dataset.TYPE_CICIDS,
                file=File(f, name="CICIDS2017_sample.csv"),
                uploaded_by=admin_user,
                num_records=1000,
                num_features=10,
                label_column="Label",
                class_distribution={"BENIGN": 700, "DDoS": 75, "PortScan": 75, "ZERO_DAY_UNKNOWN": 150},
                feature_names=['Flow Duration', 'Total Fwd Packets', 'Total Backward Packets', 'Total Length of Fwd Packets', 'Total Length of Bwd Packets', 'Flow Bytes/s', 'Flow Packets/s', 'Fwd Packet Length Mean', 'Bwd Packet Length Mean', 'Packet Length Mean'],
                status=Dataset.STATUS_READY,
                file_size_kb=round(os.path.getsize(sample_csv) / 1024.0, 2)
            )

            graph_config = GraphConfig.objects.create(
                dataset=dataset,
                name="Standard k-NN (k=5, Chunk=100)",
                graph_type=GraphConfig.GRAPH_KNN,
                k_neighbors=5,
                chunk_size=100
            )

            DatasetSplit.objects.create(
                dataset=dataset,
                train_ratio=0.70,
                val_ratio=0.15,
                test_ratio=0.15,
                train_records=700,
                val_records=150,
                test_records=150,
                is_leakage_prevented=True
            )
            print(f"[+] Ingested Dataset: '{dataset.name}' with GraphConfig #{graph_config.id}")

            # 3. Train Initial GNN Model
            print("[*] Training baseline GraphSAGE model for immediate demonstration...")
            job = TrainingJob.objects.create(
                name="GraphSAGE Self-Supervised Baseline",
                dataset=dataset,
                graph_config=graph_config,
                epochs=10,
                learning_rate=0.005,
                hidden_dim=64,
                embedding_dim=32,
                created_by=admin_user,
                status=TrainingJob.STATUS_PENDING
            )
            run_self_supervised_training(job.id)
            job.refresh_from_db()
            print(f"[+] Model Training Complete! Job #{job.id} - F1: {job.f1_score}, Loss: {job.final_loss}")

            # 4. Run Demonstration Traffic Scan
            uninspected_csv = os.path.join(base_dir, 'sample_data', 'uninspected_traffic.csv')
            if os.path.exists(uninspected_csv):
                with open(uninspected_csv, 'rb') as uf:
                    upload = TrafficUpload.objects.create(
                        user=admin_user,
                        file=File(uf, name="uninspected_traffic.csv"),
                        file_name="uninspected_traffic.csv",
                        model_used=job,
                        status=TrafficUpload.STATUS_PENDING
                    )
                    run_zero_day_detection(upload.id)
                    upload.refresh_from_db()
                    print(f"[+] Demonstration Traffic Scan Complete! Scan #{upload.id} Verdict: {upload.get_overall_verdict_display()}")

    print("[+] Database seeding completed successfully!")

if __name__ == '__main__':
    seed()
