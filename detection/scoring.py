"""
Anomaly Scoring Algorithms for Zero-Day GraphIDS.
Implements Mahalanobis distance with covariance shrinkage,
Centroid distance, and reconstruction error metrics.
"""

from typing import Dict, Any
import numpy as np
from sklearn.covariance import LedoitWolf

def compute_normal_reference(train_embeddings: np.ndarray, reg_epsilon: float = 1e-4) -> Dict[str, Any]:
    """
    Constructs the baseline normal profile from benign training graph embeddings:
    1. Computes empirical centroid mean vector.
    2. Estimates well-conditioned covariance matrix using Ledoit-Wolf shrinkage.
    3. Inverts the covariance matrix with ridge regularization for numerical stability.
    """
    if len(train_embeddings) == 0:
        raise ValueError("Cannot construct normal reference from empty training embeddings.")

    n_samples, dim = train_embeddings.shape
    centroid = np.mean(train_embeddings, axis=0)

    if n_samples > dim and n_samples > 5:
        try:
            lw = LedoitWolf()
            lw.fit(train_embeddings)
            cov = lw.covariance_
            # Ridge regularization
            cov_reg = cov + (reg_epsilon * np.eye(dim))
            cov_inv = np.linalg.pinv(cov_reg)
        except Exception:
            # Fallback to diagonal covariance
            variances = np.var(train_embeddings, axis=0) + reg_epsilon
            cov_inv = np.diag(1.0 / variances)
    else:
        # Fallback to regularized empirical variance
        variances = np.var(train_embeddings, axis=0) + reg_epsilon
        cov_inv = np.diag(1.0 / variances)

    return {
        'centroid': centroid,
        'cov_inv': cov_inv,
        'n_samples': n_samples,
        'dimension': dim
    }

def calculate_anomaly_scores(
    embeddings: np.ndarray,
    ref_model: Dict[str, Any],
    method: str = 'mahalanobis'
) -> np.ndarray:
    """
    Calculates anomaly scores for test graph embeddings against the normal reference model.
    Higher score indicates higher anomaly confidence.
    """
    if len(embeddings) == 0:
        return np.empty(0, dtype=np.float32)

    centroid = ref_model['centroid']
    cov_inv = ref_model['cov_inv']

    diff = embeddings - centroid  # [N, D]

    if method == 'mahalanobis':
        # D_M = sqrt(diff * Cov_inv * diff^T)
        left = np.dot(diff, cov_inv)  # [N, D]
        dist_sq = np.sum(left * diff, axis=1)  # [N]
        dist_sq = np.maximum(dist_sq, 0.0)
        scores = np.sqrt(dist_sq)
    elif method == 'euclidean':
        scores = np.linalg.norm(diff, axis=1)
    elif method == 'cosine':
        norms_emb = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
        norm_cent = np.linalg.norm(centroid) + 1e-8
        sim = np.dot(embeddings / norms_emb, centroid / norm_cent)
        scores = 1.0 - sim
    else:
        scores = np.linalg.norm(diff, axis=1)

    return np.nan_to_num(scores, nan=0.0, posinf=1e5, neginf=0.0).astype(np.float32)
