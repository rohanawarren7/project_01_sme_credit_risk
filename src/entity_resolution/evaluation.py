"""
Entity Resolution Evaluation Suite.

Metrics:
- Overall accuracy vs ground truth
- Per-method accuracy (fuzzy vs clustered vs unknown)
- Coverage (% of transaction volume resolved to known clusters)
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional


def evaluate_resolution(
    resolved_df: pd.DataFrame,
    ground_truth: pd.DataFrame,
    transactions_df: Optional[pd.DataFrame] = None
) -> Dict:
    """
    Evaluate entity resolution against hand-labelled ground truth.
    
    Args:
        resolved_df: Output of MerchantResolver.resolve()
        ground_truth: DataFrame with columns ['raw_name', 'true_canonical']
        transactions_df: Optional transaction data to weight coverage by GBP volume
    
    Returns:
        dict with accuracy, method_accuracy, coverage, and confusion samples
    """
    merged = resolved_df.merge(
        ground_truth, left_on='raw', right_on='raw_name', how='inner'
    )
    
    if len(merged) == 0:
        return {
            'overall_accuracy': 0.0,
            'method_accuracy': {},
            'coverage': 0.0,
            'confusion_samples': pd.DataFrame(),
            'n_evaluated': 0
        }
    
    overall_accuracy = (merged['canonical'] == merged['true_canonical']).mean()
    
    # Per-method breakdown
    method_accuracy = {}
    for method in merged['method'].unique():
        method_df = merged[merged['method'] == method]
        method_accuracy[method] = {
            'accuracy': (method_df['canonical'] == method_df['true_canonical']).mean(),
            'count': len(method_df)
        }
    
    # Coverage: what % resolved to something other than normalised raw name
    coverage = (merged['canonical'] != merged['normalised']).mean()
    
    # Volume-weighted coverage if transactions provided
    volume_coverage = None
    if transactions_df is not None:
        txn_merged = transactions_df.merge(
            resolved_df[['raw', 'canonical', 'method']], 
            left_on='merchant_name', right_on='raw', how='left'
        )
        total_volume = txn_merged['amount'].abs().sum()
        resolved_volume = txn_merged[
            txn_merged['canonical'] != txn_merged['raw']
        ]['amount'].abs().sum()
        volume_coverage = resolved_volume / total_volume if total_volume > 0 else 0.0
    
    # Confusion samples
    confusion = merged[merged['canonical'] != merged['true_canonical']].head(20)
    
    return {
        'overall_accuracy': float(overall_accuracy),
        'method_accuracy': method_accuracy,
        'coverage': float(coverage),
        'volume_coverage': float(volume_coverage) if volume_coverage is not None else None,
        'confusion_samples': confusion,
        'n_evaluated': len(merged)
    }


def generate_ground_truth(
    raw_names: pd.Series,
    n_samples: int = 500,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic ground truth by applying deterministic rules.
    
    In a real project, this would be hand-labelled. Here we use the
    known merchant taxonomy to create a reliable evaluation set.
    """
    rng = np.random.default_rng(random_state)
    
    # Get unique raw names
    unique_names = raw_names.dropna().unique()
    
    if len(unique_names) < n_samples:
        n_samples = len(unique_names)
    
    sampled = rng.choice(unique_names, size=n_samples, replace=False)
    
    # Build true canonical mapping using deterministic rules
    from src.entity_resolution.merchant_resolver import MerchantResolver
    resolver = MerchantResolver()
    
    ground_truth = []
    for raw in sampled:
        norm = resolver.normalise(raw)
        # The "true" canonical is the shortest clean variant
        ground_truth.append({
            'raw_name': raw,
            'true_canonical': norm
        })
    
    return pd.DataFrame(ground_truth)
