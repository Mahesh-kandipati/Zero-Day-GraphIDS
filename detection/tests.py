import numpy as np
from django.test import TestCase
from detection.scoring import compute_normal_reference, calculate_anomaly_scores
from detection.thresholds import calibrate_detection_threshold

class ScoringAndThresholdsTestCase(TestCase):
    def setUp(self):
        np.random.seed(42)
        # 50 normal embeddings of dimension 16
        self.normal_embeds = np.random.normal(loc=0.0, scale=1.0, size=(50, 16)).astype(np.float32)
        # 10 anomalous embeddings with shifted mean
        self.anom_embeds = np.random.normal(loc=5.0, scale=1.0, size=(10, 16)).astype(np.float32)

    def test_compute_normal_reference(self):
        ref = compute_normal_reference(self.normal_embeds)
        self.assertIn('centroid', ref)
        self.assertIn('cov_inv', ref)
        self.assertEqual(ref['dimension'], 16)
        self.assertEqual(ref['cov_inv'].shape, (16, 16))

    def test_mahalanobis_anomaly_scoring(self):
        ref = compute_normal_reference(self.normal_embeds)
        normal_scores = calculate_anomaly_scores(self.normal_embeds, ref, method='mahalanobis')
        anom_scores = calculate_anomaly_scores(self.anom_embeds, ref, method='mahalanobis')

        self.assertEqual(len(normal_scores), 50)
        self.assertEqual(len(anom_scores), 10)
        # Anomalies should receive substantially higher scores
        self.assertTrue(np.mean(anom_scores) > np.mean(normal_scores))

    def test_calibrate_threshold(self):
        ref = compute_normal_reference(self.normal_embeds)
        normal_scores = calculate_anomaly_scores(self.normal_embeds, ref)
        thresh_info = calibrate_detection_threshold(normal_scores, contamination=0.05)

        self.assertIn('threshold', thresh_info)
        self.assertTrue(thresh_info['threshold'] > 0.0)
