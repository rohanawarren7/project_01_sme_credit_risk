"""
Credit Risk Evaluation Metrics.

Comprehensive metric suite for PD model evaluation.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, brier_score_loss


def ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute KS statistic for credit risk discrimination."""
    pos_probs = y_prob[y_true == 1]
    neg_probs = y_prob[y_true == 0]
    if len(pos_probs) == 0 or len(neg_probs) == 0:
        return 0.0
    return stats.ks_2samp(pos_probs, neg_probs).statistic


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (y_prob >= bin_boundaries[i]) & (y_prob <= bin_boundaries[i + 1])
        else:
            mask = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
        
        if mask.sum() > 0:
            avg_confidence = y_prob[mask].mean()
            avg_accuracy = y_true[mask].mean()
            ece += mask.sum() / len(y_prob) * abs(avg_accuracy - avg_confidence)
    
    return ece


def decile_analysis(y_true: np.ndarray, y_prob: np.ndarray) -> pd.DataFrame:
    """Traditional credit risk decile table."""
    df = pd.DataFrame({'y': y_true, 'prob': y_prob})
    df['decile'] = pd.qcut(df['prob'], 10, labels=range(1, 11), duplicates='drop')
    
    return df.groupby('decile').agg(
        count=('y', 'count'),
        default_rate=('y', 'mean'),
        avg_pd=('prob', 'mean'),
        min_pd=('prob', 'min'),
        max_pd=('prob', 'max')
    ).reset_index()


def compute_credit_risk_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Comprehensive metric suite for PD model evaluation."""
    metrics = {}
    
    # Discrimination
    metrics['auc_roc'] = roc_auc_score(y_true, y_prob)
    metrics['gini'] = 2 * metrics['auc_roc'] - 1
    metrics['ks_statistic'] = ks_statistic(y_true, y_prob)
    
    # Calibration
    metrics['brier_score'] = brier_score_loss(y_true, y_prob)
    metrics['ece'] = expected_calibration_error(y_true, y_prob)
    
    # Decile stability
    metrics['decile_stats'] = decile_analysis(y_true, y_prob)
    
    return metrics
