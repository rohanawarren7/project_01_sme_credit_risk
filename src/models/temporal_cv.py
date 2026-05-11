"""
Temporal Cross-Validation for Credit Risk.

Time-series aware CV: train on past, test on future.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator


class TemporalGroupKFold(BaseCrossValidator):
    """
    Time-series cross-validation for credit risk.
    Splits are defined by application_date (loan origination date).
    Training: all loans originated before a cutoff.
    Testing: loans originated in the next window.
    """
    
    def __init__(self, n_splits: int = 5, test_size: float = 0.2,
                 date_col: str = 'application_date'):
        self.n_splits = n_splits
        self.test_size = test_size
        self.date_col = date_col
    
    def split(self, X, y=None, groups=None):
        dates = pd.to_datetime(X[self.date_col])
        sorted_idx = np.argsort(dates)
        n_samples = len(dates)
        
        # Calculate test size in samples
        test_n = int(n_samples * self.test_size)
        
        for i in range(self.n_splits):
            # Calculate split point
            test_start = n_samples - (self.n_splits - i) * test_n
            test_end = min(test_start + test_n, n_samples)
            
            if test_start <= 0 or test_end <= test_start:
                continue
            
            train_idx = sorted_idx[:test_start]
            test_idx = sorted_idx[test_start:test_end]
            
            yield train_idx, test_idx
    
    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits
