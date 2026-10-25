"""
Command-Line GNN Model Trainer for Zero-Day GraphIDS.
Enables standalone training and checkpoint saving without web interface.
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
import torch

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_engine.graph_builder import GraphBuilder
from ml_engine.encoder import GraphSAGEEncoder
from ml_engine.self_supervised import SelfSupervisedGraphLoss
from ml_engine.embeddings import extract_chunk_embeddings
from ml_engine.checkpoints import save_checkpoint
from datasets.preprocessing import clean_and_sanitize_flow_data, split_and_normalize_dataset
from detection.scoring import compute_normal_reference, calculate_anomaly_scores
from detection.thresholds import calibrate_detection_threshold
from training.evaluation import evaluate_detector_predictions

def train_cli(
    csv_path: str,
    output_model_path: str,
    epochs: int = 15,
    lr: float = 0.005,
    hidden_dim: int = 64,
    embedding_dim: int = 32,
    chunk_size: int = 100,
    k_neighbors: int = 5
):
    print("=" * 70)
    print("   ZERO-DAY GRAPHIDS // GNN SELF-SUPERVISED TRAINER")
    print("=" * 70)
    print(f"[*] Reading dataset: {csv_path}")

    df_raw = pd.read_csv(csv_path)
    df_clean, label_col, feature_cols = clean_and_sanitize_flow_data(df_raw)
    print(f"[+] Loaded {len(df_clean)} flows with {len(feature_cols)} features. Ground-truth: '{label_col}'")

    # Split dataset
    split = split_and_normalize_dataset(df_clean, label_col=label_col, train_ratio=0.7, val_ratio=0.15)
    scaler = split['scaler']

    X_train = split['train']['X']
    y_train = split['train']['y']
    X_val = split['val']['X']
    y_val = split['val']['y']
    X_test = split['test']['X']
    y_test = split['test']['y']

    # Filter training to normal flows only
    if label_col:
        normal_mask = np.array([str(lbl).strip().upper() in ('BENIGN', 'NORMAL', '0') for lbl in y_train])
        if np.sum(normal_mask) >= 10:
            X_train = X_train[normal_mask]
            print(f"[+] Strictly filtered training set to {len(X_train)} normal flows for self-supervision.")

    # Build Graph Chunks
    builder = GraphBuilder(k_neighbors=k_neighbors)
    train_chunks = builder.build_flow_chunks(X_train, chunk_size=chunk_size)
    val_chunks = builder.build_flow_chunks(X_val, chunk_size=chunk_size)
    test_chunks = builder.build_flow_chunks(X_test, chunk_size=chunk_size)
    print(f"[+] Constructed {len(train_chunks)} training chunks, {len(val_chunks)} validation chunks.")

    # Model and Loss
    in_dim = len(feature_cols)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[*] Compute Device: {device.upper()}")

    encoder = GraphSAGEEncoder(in_dim=in_dim, hidden_dim=hidden_dim, embedding_dim=embedding_dim).to(device)
    criterion = SelfSupervisedGraphLoss(node_dim=embedding_dim, graph_dim=encoder.graph_dim).to(device)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(criterion.parameters()), lr=lr)

    # Training Loop
    print("[*] Starting self-supervised optimization...")
    for epoch in range(1, epochs + 1):
        encoder.train()
        total_epoch_loss = 0.0

        for chunk in train_chunks:
            data = chunk['pyg_data']
            x = data.x.to(device)
            edge_index = data.edge_index.to(device)

            if x.size(0) == 0:
                continue

            optimizer.zero_grad()
            loss, metrics = criterion(encoder, x, edge_index)
            loss.backward()
            optimizer.step()
            total_epoch_loss += metrics['total_loss']

        avg_loss = total_epoch_loss / max(1, len(train_chunks))
        if epoch % 5 == 0 or epoch == epochs:
            print(f"    Epoch {epoch:02d}/{epochs:02d} | Self-Supervised Loss: {avg_loss:.4f}")

    # Reference Model & Threshold Calibration
    print("[*] Constructing reference normal manifold and calibrating threshold...")
    train_embeds, _ = extract_chunk_embeddings(encoder, train_chunks, device=device)
    ref_model = compute_normal_reference(train_embeds)

    val_embeds, _ = extract_chunk_embeddings(encoder, val_chunks, device=device)
    val_scores = calculate_anomaly_scores(val_embeds, ref_model, method='mahalanobis')
    threshold_info = calibrate_detection_threshold(val_scores, contamination=0.03)
    thresh = threshold_info['threshold']
    print(f"[+] Calibrated Anomaly Threshold: {thresh:.4f}")

    # Evaluate on Test Set
    test_embeds, _ = extract_chunk_embeddings(encoder, test_chunks, device=device)
    test_scores = calculate_anomaly_scores(test_embeds, ref_model, method='mahalanobis')
    test_preds = (test_scores > thresh).astype(int)

    chunk_ground_truth = []
    for c in test_chunks:
        c_lbls = c['labels']
        is_mal = any(str(l).strip().upper() not in ('BENIGN', 'NORMAL', '0') for l in c_lbls) if c_lbls is not None else False
        chunk_ground_truth.append(1 if is_mal else 0)

    metrics = evaluate_detector_predictions(np.array(chunk_ground_truth), test_preds, test_scores)
    print(f"[+] Test Evaluation: F1-Score: {metrics['f1_score']:.3f} | ROC-AUC: {metrics['roc_auc']:.3f} | Accuracy: {metrics['accuracy']:.3f}")

    # Save Checkpoint
    scaler_params = {
        'mean_': scaler.mean_.tolist(),
        'scale_': scaler.scale_.tolist(),
        'feature_names': feature_cols
    }
    save_checkpoint(
        file_path=output_model_path,
        encoder=encoder,
        scaler_params=scaler_params,
        threshold_info=threshold_info,
        metadata={
            'epochs': epochs,
            'final_loss': avg_loss,
            'f1_score': metrics['f1_score'],
            'ref_centroid': ref_model['centroid'].tolist(),
            'ref_cov_inv': ref_model['cov_inv'].tolist()
        }
    )
    print(f"[+] Saved model checkpoint to: {output_model_path}")
    print("=" * 70)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Zero-Day GraphIDS Standalone Trainer")
    parser.add_argument('--dataset', type=str, default='sample_data/CICIDS2017_sample.csv')
    parser.add_argument('--output', type=str, default='model_artifacts/gnn_model.pt')
    parser.add_argument('--epochs', type=int, default=15)
    args = parser.parse_args()

    train_cli(csv_path=args.dataset, output_model_path=args.output, epochs=args.epochs)
