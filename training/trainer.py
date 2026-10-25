"""
Self-Supervised GNN Model Trainer for Zero-Day GraphIDS.
Executes optimization loops, validation checks, and checkpoint creation.
"""

import os
import time
import numpy as np
import pandas as pd
import torch
from django.conf import settings
from .models import TrainingJob, TrainingMetric
from ml_engine.graph_builder import GraphBuilder
from ml_engine.encoder import GraphSAGEEncoder
from ml_engine.self_supervised import SelfSupervisedGraphLoss
from ml_engine.embeddings import extract_chunk_embeddings
from ml_engine.checkpoints import save_checkpoint
from datasets.preprocessing import clean_and_sanitize_flow_data, split_and_normalize_dataset
from detection.scoring import compute_normal_reference, calculate_anomaly_scores
from detection.thresholds import calibrate_detection_threshold
from training.evaluation import evaluate_detector_predictions

def run_self_supervised_training(job_id: int) -> TrainingJob:
    """
    Main training execution function for TrainingJob.
    Trains GraphSAGE encoder strictly on normal flows via self-supervision,
    calibrates anomaly threshold on validation flows, and evaluates test flows.
    """
    job = TrainingJob.objects.get(pk=job_id)
    job.status = TrainingJob.STATUS_RUNNING
    job.save()

    start_time = time.time()
    log_messages = []

    def log(msg: str):
        log_messages.append(f"[{time.strftime('%X')}] {msg}")
        job.logs = "\n".join(log_messages)
        job.save(update_fields=['logs'])

    try:
        log(f"Initiating training run for job '{job.name}' (ID: {job.id}).")
        dataset = job.dataset
        graph_config = job.graph_config

        # 1. Ingest Dataset
        csv_path = dataset.file.path
        log(f"Reading dataset CSV from: {csv_path}")
        df_raw = pd.read_csv(csv_path, low_memory=False)

        # 2. Clean & Sanitize
        df_clean, label_col, feature_cols = clean_and_sanitize_flow_data(df_raw)
        log(f"Identified {len(df_clean)} flows with {len(feature_cols)} features. Label column: '{label_col}'.")

        # 3. Split Dataset
        split = split_and_normalize_dataset(df_clean, label_col=label_col, train_ratio=0.70, val_ratio=0.15)
        scaler = split['scaler']

        X_train = split['train']['X']
        y_train = split['train']['y']
        X_val = split['val']['X']
        y_val = split['val']['y']
        X_test = split['test']['X']
        y_test = split['test']['y']

        # 4. Strict Self-Supervised Paradigm: Filter training set to NORMAL / BENIGN flows only!
        # This guarantees zero label leakage during representation learning.
        if label_col:
            normal_mask = np.array([str(lbl).strip().upper() in ('BENIGN', 'NORMAL', '0') for lbl in y_train])
            if np.sum(normal_mask) >= 10:
                X_train = X_train[normal_mask]
                y_train = y_train[normal_mask]
                log(f"Filtered training set to {len(X_train)} verified normal flows for self-supervision.")
            else:
                log("Warning: Limited normal flows in training split; utilizing all available training flows.")

        # 5. Build Graph Chunks
        builder = GraphBuilder(k_neighbors=graph_config.k_neighbors, metric=graph_config.similarity_metric)
        train_chunks = builder.build_flow_chunks(X_train, chunk_size=graph_config.chunk_size)
        val_chunks = builder.build_flow_chunks(X_val, chunk_size=graph_config.chunk_size)
        test_chunks = builder.build_flow_chunks(X_test, chunk_size=graph_config.chunk_size)

        log(f"Constructed {len(train_chunks)} training graph chunks, {len(val_chunks)} validation chunks.")

        # 6. Initialize GNN & Self-Supervised Loss
        in_dim = len(feature_cols)
        device = 'cuda' if torch.cuda.is_available() and getattr(settings, 'ML_DEVICE', 'cpu') == 'cuda' else 'cpu'
        log(f"Initializing GraphSAGE encoder on compute device: {device.upper()}")

        encoder = GraphSAGEEncoder(
            in_dim=in_dim,
            hidden_dim=job.hidden_dim,
            embedding_dim=job.embedding_dim,
            dropout=0.1
        ).to(device)

        criterion = SelfSupervisedGraphLoss(
            node_dim=job.embedding_dim,
            graph_dim=encoder.graph_dim,
            contrastive_weight=0.5
        ).to(device)

        optimizer = torch.optim.Adam(
            list(encoder.parameters()) + list(criterion.parameters()),
            lr=job.learning_rate,
            weight_decay=1e-5
        )

        # 7. Training Loop
        epochs = job.epochs
        best_loss = float('inf')

        for epoch in range(1, epochs + 1):
            encoder.train()
            epoch_loss = 0.0
            epoch_recon = 0.0
            epoch_contra = 0.0

            # Shuffle chunks
            perm = np.random.permutation(len(train_chunks))
            for idx in perm:
                chunk = train_chunks[idx]
                pyg_data = chunk['pyg_data']
                x = pyg_data.x.to(device)
                edge_index = pyg_data.edge_index.to(device)

                if x.size(0) == 0:
                    continue

                optimizer.zero_grad()
                loss, metrics = criterion(encoder, x, edge_index)
                loss.backward()
                optimizer.step()

                epoch_loss += metrics['total_loss']
                epoch_recon += metrics['recon_loss']
                epoch_contra += metrics['contrastive_loss']

            n_chunks = max(1, len(train_chunks))
            avg_train_loss = epoch_loss / n_chunks
            avg_recon = epoch_recon / n_chunks
            avg_contra = epoch_contra / n_chunks

            # Compute validation loss
            encoder.eval()
            val_loss = 0.0
            with torch.no_grad():
                for v_chunk in val_chunks:
                    v_data = v_chunk['pyg_data']
                    vx = v_data.x.to(device)
                    ve = v_data.edge_index.to(device)
                    v_l, _ = criterion(encoder, vx, ve)
                    val_loss += v_l.item()
            avg_val_loss = val_loss / max(1, len(val_chunks))

            # Record metric in DB
            TrainingMetric.objects.create(
                job=job,
                epoch=epoch,
                train_loss=round(avg_train_loss, 4),
                recon_loss=round(avg_recon, 4),
                contrastive_loss=round(avg_contra, 4),
                val_loss=round(avg_val_loss, 4)
            )

            if avg_val_loss < best_loss:
                best_loss = avg_val_loss

            if epoch % 5 == 0 or epoch == epochs:
                log(f"Epoch {epoch}/{epochs} - Train Loss: {avg_train_loss:.4f} (Recon: {avg_recon:.4f}, Contra: {avg_contra:.4f}) | Val Loss: {avg_val_loss:.4f}")

        # 8. Extract Reference Normal Embeddings & Calibrate Threshold
        log("Extracting learned representations and calibrating zero-day anomaly threshold...")
        train_embeds, _ = extract_chunk_embeddings(encoder, train_chunks, device=device)
        ref_model = compute_normal_reference(train_embeds)

        val_embeds, _ = extract_chunk_embeddings(encoder, val_chunks, device=device)
        val_scores = calculate_anomaly_scores(val_embeds, ref_model, method='mahalanobis')
        threshold_info = calibrate_detection_threshold(val_scores, contamination=0.03)
        calibrated_threshold = threshold_info['threshold']
        log(f"Calibrated anomaly detection threshold: {calibrated_threshold:.4f} (Method: {threshold_info['method']})")

        # 9. Evaluate on Test Set
        test_embeds, _ = extract_chunk_embeddings(encoder, test_chunks, device=device)
        test_scores = calculate_anomaly_scores(test_embeds, ref_model, method='mahalanobis')
        test_preds = (test_scores > calibrated_threshold).astype(int)

        # Evaluate ground-truth labels for chunks (chunk is malicious if >= 20% flows are malicious)
        chunk_ground_truth = []
        for c in test_chunks:
            c_lbls = c['labels']
            if c_lbls is not None and len(c_lbls) > 0:
                is_mal = any(str(l).strip().upper() not in ('BENIGN', 'NORMAL', '0') for l in c_lbls)
                chunk_ground_truth.append(1 if is_mal else 0)
            else:
                chunk_ground_truth.append(0)

        eval_metrics = evaluate_detector_predictions(np.array(chunk_ground_truth), test_preds, test_scores)

        # 10. Persist Model Checkpoint
        models_dir = os.path.join(settings.MEDIA_ROOT, 'models')
        os.makedirs(models_dir, exist_ok=True)
        model_filename = f"job_{job.id}_gnn.pt"
        checkpoint_path = os.path.join(models_dir, model_filename)

        scaler_params = {
            'mean_': scaler.mean_.tolist(),
            'scale_': scaler.scale_.tolist(),
            'feature_names': feature_cols
        }

        save_checkpoint(
            file_path=checkpoint_path,
            encoder=encoder,
            scaler_params=scaler_params,
            threshold_info=threshold_info,
            metadata={
                'job_id': job.id,
                'epochs': epochs,
                'final_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'ref_centroid': ref_model['centroid'].tolist(),
                'ref_cov_inv': ref_model['cov_inv'].tolist()
            }
        )

        job.model_file = f"models/{model_filename}"
        job.final_loss = round(avg_train_loss, 4)
        job.accuracy = eval_metrics['accuracy']
        job.precision = eval_metrics['precision']
        job.recall = eval_metrics['recall']
        job.f1_score = eval_metrics['f1_score']
        job.roc_auc = eval_metrics['roc_auc']
        job.pr_auc = eval_metrics['pr_auc']
        job.status = TrainingJob.STATUS_COMPLETED
        job.training_duration_sec = round(time.time() - start_time, 2)
        job.completed_at = pd.Timestamp.now()
        job.save()

        log(f"Training run finished successfully in {job.training_duration_sec}s. Artifact saved to: {job.model_file.name}")
        return job

    except Exception as e:
        job.status = TrainingJob.STATUS_FAILED
        job.error_message = str(e)
        job.training_duration_sec = round(time.time() - start_time, 2)
        job.save()
        log(f"Training execution encountered an error: {e}")
        raise e
