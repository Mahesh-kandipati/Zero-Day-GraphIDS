import numpy as np
from django.test import TestCase
from training.evaluation import evaluate_detector_predictions, run_baseline_comparison

class EvaluationFrameworkTestCase(TestCase):
    def setUp(self):
        np.random.seed(42)
        # Synthetic binary evaluation labels: 0=normal, 1=zero-day attack
        self.y_true = np.array([0]*80 + [1]*20)
        # Synthetic anomaly scores: higher for attacks
        self.scores = np.concatenate([
            np.random.normal(loc=0.2, scale=0.1, size=80),
            np.random.normal(loc=0.8, scale=0.1, size=20)
        ])
        threshold = 0.5
        self.y_pred = (self.scores > threshold).astype(int)

    def test_evaluate_detector_predictions(self):
        metrics = evaluate_detector_predictions(self.y_true, self.y_pred, self.scores)
        self.assertIn('accuracy', metrics)
        self.assertIn('f1_score', metrics)
        self.assertIn('roc_auc', metrics)
        self.assertIn('confusion_matrix', metrics)
        self.assertTrue(metrics['f1_score'] > 0.7)
        self.assertTrue(metrics['roc_auc'] > 0.8)
        self.assertEqual(metrics['confusion_matrix']['tp'] + metrics['confusion_matrix']['fn'], 20)

    def test_baseline_comparison(self):
        X_train_raw = np.random.randn(100, 5)
        X_test_raw = np.random.randn(50, 5)
        y_test = np.array(['BENIGN']*40 + ['ZERO_DAY_ANOMALY']*10)

        prop_metrics = {'f1_score': 0.85, 'roc_auc': 0.92, 'precision': 0.88, 'recall': 0.82}
        results = run_baseline_comparison(X_train_raw, X_test_raw, y_test, prop_metrics)

        self.assertIn('proposed_graphids', results)
        self.assertIn('isolation_forest', results)
        self.assertIn('raw_centroid_distance', results)
        self.assertIn('f1_score', results['isolation_forest'])
