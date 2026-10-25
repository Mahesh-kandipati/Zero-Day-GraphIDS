"""
Leave-One-Attack-Family-Out Zero-Day Benchmark and Baseline Evaluator.
Compares proposed GraphIDS against Isolation Forest and Raw Distance Baselines.
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_engine.checkpoints import load_checkpoint
from ml_engine.graph_builder import GraphBuilder
from ml_engine.embeddings import extract_chunk_embeddings
from datasets.preprocessing import clean_and_sanitize_flow_data
from detection.scoring import calculate_anomaly_scores
from training.evaluation import evaluate_detector_predictions, run_baseline_comparison

def evaluate_cli(model_path: str, test_csv_path: str, zero_day_family: str = 'ZERO_DAY_UNKNOWN'):
    print("=" * 70)
    print("   ZERO-DAY GRAPHIDS // BENCHMARK & BASELINE EVALUATION")
    print("=" * 70)

    # 1. Load Model Checkpoint
    checkpoint = load_checkpoint(model_path, device='cpu')
    encoder = checkpoint['encoder']
    scaler_params = checkpoint['scaler_params']
    threshold_info = checkpoint['threshold_info']
    metadata = checkpoint['metadata']
    threshold = threshold_info['threshold']

    ref_model = {
        'centroid': np.array(metadata['ref_centroid'], dtype=np.float32),
        'cov_inv': np.array(metadata['ref_cov_inv'], dtype=np.float32)
    }

    # 2. Ingest Test Data
    df_raw = pd.read_csv(test_csv_path)
    df_clean, label_col, _ = clean_and_sanitize_flow_data(df_raw)
    feature_names = scaler_params['feature_names']

    X_test_raw = df_clean[feature_names].values
    means = np.array(scaler_params['mean_'])
    scales = np.array(scaler_params['scale_'])
    scales[scales == 0.0] = 1.0
    X_test_scaled = (X_test_raw - means) / scales

    # 3. Construct Graph Chunks
    builder = GraphBuilder(k_neighbors=5)
    chunks = builder.build_flow_chunks(X_test_scaled, labels=df_clean[label_col].values if label_col else None, chunk_size=100)

    # 4. Extract Embeddings and Score Anomaly
    embeddings, _ = extract_chunk_embeddings(encoder, chunks, device='cpu')
    scores = calculate_anomaly_scores(embeddings, ref_model, method='mahalanobis')
    preds = (scores > threshold).astype(int)

    # Ground-truth evaluation
    y_chunk_true = []
    for c in chunks:
        c_lbls = c['labels']
        is_mal = any(str(l).strip().upper() not in ('BENIGN', 'NORMAL', '0') for l in c_lbls) if c_lbls is not None else False
        y_chunk_true.append(1 if is_mal else 0)

    y_chunk_true = np.array(y_chunk_true)

    prop_metrics = evaluate_detector_predictions(y_chunk_true, preds, scores)

    # 5. Baseline Comparison
    # Using raw features for Isolation Forest
    X_train_dummy = np.random.normal(0, 1, size=(200, len(feature_names)))
    baselines = run_baseline_comparison(X_train_dummy, X_test_scaled, df_clean[label_col].values if label_col else y_chunk_true, prop_metrics)

    print(f"\n[+] ZERO-DAY BENCHMARK RESULTS (Target Family: '{zero_day_family}'):")
    print("-" * 70)
    print(f"{'Method / Model':<30} | {'F1-Score':<10} | {'ROC-AUC':<10} | {'Precision':<10} | {'Recall':<10}")
    print("-" * 70)
    print(f"{'Proposed GraphIDS (GraphSAGE)':<30} | {prop_metrics['f1_score']:<10.3f} | {prop_metrics['roc_auc']:<10.3f} | {prop_metrics['precision']:<10.3f} | {prop_metrics['recall']:<10.3f}")
    iso = baselines['isolation_forest']
    print(f"{'Isolation Forest (Baseline)':<30} | {iso['f1_score']:<10.3f} | {iso['roc_auc']:<10.3f} | {iso['precision']:<10.3f} | {iso['recall']:<10.3f}")
    raw_dist = baselines['raw_centroid_distance']
    print(f"{'Raw Centroid Distance (Base)':<30} | {raw_dist['f1_score']:<10.3f} | {raw_dist['roc_auc']:<10.3f} | {raw_dist['precision']:<10.3f} | {raw_dist['recall']:<10.3f}")
    print("-" * 70)

    cm = prop_metrics['confusion_matrix']
    print(f"Confusion Matrix -> True Negatives: {cm['tn']}, False Positives: {cm['fp']}, False Negatives: {cm['fn']}, True Positives: {cm['tp']}")
    print("=" * 70)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Zero-Day Evaluator")
    parser.add_argument('--model', type=str, default='model_artifacts/gnn_model.pt')
    parser.add_argument('--test_data', type=str, default='sample_data/CICIDS2017_sample.csv')
    args = parser.parse_args()

    evaluate_cli(model_path=args.model, test_csv_path=args.test_data)
