import os
import tempfile
import pandas as pd
from datetime import datetime
from django.test import TestCase
from accounts.models import CustomUser
from reports.report_generator import generate_pdf_training_report

class ReportsGeneratorTestCase(TestCase):
    def setUp(self):
        self.user = CustomUser.objects.create_user(
            username='reportauditor',
            password='AuditorPassword123!',
            role=CustomUser.ROLE_ANALYST
        )

    def test_pdf_training_report_compilation(self):
        # Mock training job class
        class MockJob:
            id = 99
            name = "Mock Self-Supervised Job"
            epochs = 10
            learning_rate = 0.005
            embedding_dim = 32
            final_loss = 0.045
            f1_score = 0.92
            roc_auc = 0.98
            training_duration_sec = 12.5
            created_at = datetime.now()
            class dataset:
                name = "Mock CIC-IDS2017"
            class graph_config:
                name = "Mock k-NN (k=5)"

        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = os.path.join(tmpdir, "test_audit_report.pdf")
            generate_pdf_training_report(MockJob, pdf_path)
            self.assertTrue(os.path.exists(pdf_path))
            self.assertTrue(os.path.getsize(pdf_path) > 1000)
