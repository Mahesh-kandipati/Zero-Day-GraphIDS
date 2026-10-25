"""
End-to-End Zero-Day Detection Inference Pipeline.
Loads trained GNN checkpoints, processes uninspected traffic CSVs,
extracts graph representations, scores anomalies, and classifies severity.
"""

import time
import os
import numpy as np
import pandas as pd
import torch
from django.conf import settings
from .models import TrafficUpload, DetectionResult, AlertRecord
from ml_engine.checkpoints import load_checkpoint
from ml_engine.graph_builder import GraphBuilder
from ml_engine.embeddings import extract_chunk_embeddings
from datasets.preprocessing import clean_and_sanitize_flow_data
from .scoring import calculate_anomaly_scores

def run_zero_day_detection(upload_id: int) -> TrafficUpload:
    """
    Executes graph inference on user-uploaded traffic CSV.
    """
    upload = TrafficUpload.objects.get(pk=upload_id)
    upload.status = TrafficUpload.STATUS_ANALYZING
    upload.save()

    start_time = time.time()

    try:
        job = upload.model_used
        if not job.model_file or not os.path.exists(job.model_file.path):
            raise FileNotFoundError(f"Model file for job #{job.id} does not exist.")

        # 1. Load GNN Model & Checkpoint Context
        device = 'cuda' if torch.cuda.is_available() and getattr(settings, 'ML_DEVICE', 'cpu') == 'cuda' else 'cpu'
        checkpoint_data = load_checkpoint(job.model_file.path, device=device)
        encoder = checkpoint_data['encoder']
        scaler_params = checkpoint_data['scaler_params']
        threshold_info = checkpoint_data['threshold_info']
        metadata = checkpoint_data['metadata']

        threshold = float(threshold_info.get('threshold', 1.0))
        upload.threshold_used = threshold

        # Reference Model
        ref_model = {
            'centroid': np.array(metadata['ref_centroid'], dtype=np.float32),
            'cov_inv': np.array(metadata['ref_cov_inv'], dtype=np.float32)
        }

        # 2. Read & Clean Uninspected Traffic CSV
        df_raw = pd.read_csv(upload.file.path, low_memory=False)
        df_clean, _, _ = clean_and_sanitize_flow_data(df_raw)

        # Scale features using saved training parameters
        feature_names = scaler_params.get('feature_names', [])
        # Align features
        for f in feature_names:
            if f not in df_clean.columns:
                df_clean[f] = 0.0

        X_test = df_clean[feature_names].values
        means = np.array(scaler_params['mean_'])
        scales = np.array(scaler_params['scale_'])
        scales[scales == 0.0] = 1.0

        X_scaled = (X_test - means) / scales
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=1e5, neginf=-1e5)

        # 3. Construct Graph Chunks
        chunk_size = job.graph_config.chunk_size
        k_neighbors = job.graph_config.k_neighbors
        builder = GraphBuilder(k_neighbors=k_neighbors, metric=job.graph_config.similarity_metric)
        chunks = builder.build_flow_chunks(X_scaled, chunk_size=chunk_size)

        if not chunks:
            upload.status = TrafficUpload.STATUS_COMPLETED
            upload.overall_verdict = TrafficUpload.VERDICT_BENIGN
            upload.save()
            return upload

        # 4. Extract Graph Embeddings
        embeddings, chunk_ids = extract_chunk_embeddings(encoder, chunks, device=device)

        # 5. Calculate Mahalanobis Anomaly Scores
        anomaly_scores = calculate_anomaly_scores(embeddings, ref_model, method='mahalanobis')

        # 6. Evaluate Chunk Verdicts and Save Records
        total_chunks = len(chunks)
        malicious_count = 0
        normal_count = 0
        score_sum = 0.0
        max_score = 0.0

        for i, chunk in enumerate(chunks):
            score = float(anomaly_scores[i])
            score_sum += score
            if score > max_score:
                max_score = score

            is_mal = score > threshold

            if not is_mal:
                attack_type = DetectionResult.ATTACK_NORMAL
                normal_count += 1
            elif score <= 1.5 * threshold:
                attack_type = DetectionResult.ATTACK_PROBE
                malicious_count += 1
            elif score <= 2.5 * threshold:
                attack_type = DetectionResult.ATTACK_INTRUSION
                malicious_count += 1
            else:
                attack_type = DetectionResult.ATTACK_ZERO_DAY
                malicious_count += 1

            det_res = DetectionResult.objects.create(
                upload=upload,
                chunk_id=chunk['chunk_id'],
                start_idx=chunk['start_idx'],
                end_idx=chunk['end_idx'],
                flow_count=chunk['size'],
                anomaly_score=round(score, 4),
                threshold=round(threshold, 4),
                is_malicious=is_mal,
                attack_type=attack_type,
                mahalanobis_dist=round(score, 4)
            )

            # Generate Alert for critical or intrusion chunks
            if is_mal and attack_type in (DetectionResult.ATTACK_ZERO_DAY, DetectionResult.ATTACK_INTRUSION):
                sev = AlertRecord.SEVERITY_CRITICAL if attack_type == DetectionResult.ATTACK_ZERO_DAY else AlertRecord.SEVERITY_HIGH
                AlertRecord.objects.create(
                    upload=upload,
                    detection_result=det_res,
                    title=f"{det_res.get_attack_type_display()} in Chunk #{chunk['chunk_id']} (Score: {score:.2f} > Thresh: {threshold:.2f})",
                    severity=sev,
                    status=AlertRecord.STATUS_NEW,
                    notes=f"Detected abnormal topological flow pattern across flows {chunk['start_idx']} - {chunk['end_idx']}."
                )

        avg_score = score_sum / max(1, total_chunks)

        # Overall Verdict Determination
        malicious_ratio = malicious_count / max(1, total_chunks)
        if malicious_ratio == 0:
            overall_verdict = TrafficUpload.VERDICT_BENIGN
        elif any(c.attack_type == DetectionResult.ATTACK_ZERO_DAY for c in upload.results.all()):
            overall_verdict = TrafficUpload.VERDICT_ZERO_DAY
        elif malicious_ratio > 0.25:
            overall_verdict = TrafficUpload.VERDICT_HIGH_RISK
        else:
            overall_verdict = TrafficUpload.VERDICT_SUSPICIOUS

        elapsed_ms = round((time.time() - start_time) * 1000.0, 2)

        upload.total_chunks = total_chunks
        upload.malicious_chunks = malicious_count
        upload.normal_chunks = normal_count
        upload.avg_anomaly_score = round(avg_score, 4)
        upload.max_anomaly_score = round(max_score, 4)
        upload.overall_verdict = overall_verdict
        upload.latency_ms = elapsed_ms
        upload.status = TrafficUpload.STATUS_COMPLETED
        upload.save()

        return upload

    except Exception as e:
        upload.status = TrafficUpload.STATUS_FAILED
        upload.error_message = str(e)
        upload.save()
        raise e
