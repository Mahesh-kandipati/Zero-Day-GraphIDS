"""
Threshold Calibration for Zero-Day Anomaly Detection.
Determines decision boundaries exclusively using validation distribution
to guarantee that test labels are never leaked into threshold selection.
"""

from typing import Dict, Any
import numpy as np

def calibrate_detection_threshold(
    val_scores: np.ndarray,
    contamination: float = 0.03,
    method: str = 'percentile'
) -> Dict[str, Any]:
    """
    Calibrates decision boundary on validation anomaly scores:
    1. 'percentile': Sets threshold at (1 - contamination) * 100 percentile.
    2. 'chebyshev': Sets threshold at mean + k * std (e.g. k=3).
    """
    if len(val_scores) == 0:
        return {'threshold': 1.0, 'method': 'default', 'contamination': contamination}

    clean_scores = np.nan_to_num(val_scores, nan=0.0)

    if method == 'chebyshev':
        mean_val = float(np.mean(clean_scores))
        std_val = float(np.std(clean_scores))
        threshold = mean_val + (3.0 * std_val)
    else:
        # Default: percentile-based adaptive cutoff
        percentile_val = (1.0 - contamination) * 100.0
        threshold = float(np.percentile(clean_scores, percentile_val))

    # Safeguard against zero threshold
    if threshold <= 1e-6:
        threshold = float(np.max(clean_scores) * 1.1) if np.max(clean_scores) > 0 else 1.0

    return {
        'threshold': round(threshold, 4),
        'method': method,
        'contamination': contamination,
        'val_mean': round(float(np.mean(clean_scores)), 4),
        'val_std': round(float(np.std(clean_scores)), 4),
        'val_max': round(float(np.max(clean_scores)), 4)
    }
