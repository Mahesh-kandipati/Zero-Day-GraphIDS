import tempfile
import pandas as pd
from django.test import TestCase
from django.core.files.uploadedfile import SimpleUploadedFile
from accounts.models import CustomUser
from datasets.models import Dataset, GraphConfig
from datasets.preprocessing import clean_and_sanitize_flow_data, split_and_normalize_dataset

class DatasetPreprocessingTestCase(TestCase):
    def setUp(self):
        self.user = CustomUser.objects.create_user(
            username='dataanalyst',
            password='TestPassword123!',
            role=CustomUser.ROLE_ANALYST
        )

        # Create synthetic tabular flow test dataframe
        self.df = pd.DataFrame({
            'Flow ID': [f'flow_{i}' for i in range(20)],
            ' Source IP': ['192.168.1.1'] * 20,
            'Flow Duration': [1000 + i * 50 for i in range(20)],
            'Total Fwd Packets': [10 + i for i in range(20)],
            'Total Length of Fwd Packets': [500 + i * 20 for i in range(20)],
            'Flow Bytes/s': [100.5 + i for i in range(20)],
            ' Label': ['BENIGN'] * 15 + ['DDoS'] * 5
        })

    def test_clean_and_sanitize_flow_data(self):
        df_clean, label_col, features = clean_and_sanitize_flow_data(self.df)
        self.assertEqual(label_col, 'Label')
        self.assertNotIn('Flow ID', df_clean.columns)
        self.assertIn('Flow Duration', features)
        self.assertIn('Total Fwd Packets', features)
        self.assertEqual(len(df_clean), 20)

    def test_split_and_normalize_leakage_free(self):
        df_clean, label_col, _ = clean_and_sanitize_flow_data(self.df)
        split = split_and_normalize_dataset(df_clean, label_col=label_col, train_ratio=0.7, val_ratio=0.15)
        self.assertIn('scaler', split)
        self.assertEqual(split['train']['X'].shape[1], len(split['feature_cols']))
        self.assertTrue(split['train']['count'] > 0)
        self.assertTrue(split['val']['count'] > 0)
        self.assertTrue(split['test']['count'] > 0)

    def test_dataset_model_creation(self):
        csv_content = self.df.to_csv(index=False).encode('utf-8')
        uploaded_file = SimpleUploadedFile("test_traffic.csv", csv_content, content_type="text/csv")

        dataset = Dataset.objects.create(
            name="Unit Test Flow Dataset",
            dataset_type=Dataset.TYPE_CICIDS,
            file=uploaded_file,
            uploaded_by=self.user,
            num_records=20
        )
        self.assertEqual(dataset.name, "Unit Test Flow Dataset")
        self.assertTrue(dataset.file.name.endswith(".csv"))
