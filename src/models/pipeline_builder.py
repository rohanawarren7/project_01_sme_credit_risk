"""
Model Pipeline Builder.

Constructs sklearn preprocessing + model pipelines.
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


# Define feature groups
NUMERIC_FEATURES = [
    'cv_monthly_net', 'net_operating_ratio', 'txn_frequency_trend',
    'max_zero_day_streak_90d', 'recurring_revenue_ratio', 'months_runway',
    'supplier_hhi', 'velocity_90_180', 'txn_count_90d', 'txn_count_180d',
    'mean_txn_amount', 'median_txn_amount', 'std_txn_amount',
    'total_inflow', 'total_outflow', 'active_days_ratio',
    'google_sentiment_mean', 'google_sentiment_trend', 'google_review_velocity',
    'trustpilot_90d_count', 'trustpilot_rating_mean',
    'headcount_growth_rate', 'headcount_latest',
    'num_officers', 'num_shareholders', 'share_capital',
    'mortgages_count', 'charges_count'
]

# Features with heavy tails
ROBUST_FEATURES = ['cv_monthly_net', 'months_runway', 'supplier_hhi', 'velocity_90_180',
                   'max_zero_day_streak_90d', 'std_txn_amount']

STANDARD_FEATURES = [f for f in NUMERIC_FEATURES if f not in ROBUST_FEATURES]


def get_available_features(X: pd.DataFrame) -> list:
    """Get features that actually exist in the data and have some non-null values."""
    available = []
    for col in NUMERIC_FEATURES:
        if col in X.columns and X[col].notna().any():
            available.append(col)
    return available


def build_preprocessing_pipeline(available_features: list) -> Pipeline:
    """Build preprocessing pipeline with imputation and scaling."""
    robust_available = [f for f in ROBUST_FEATURES if f in available_features]
    standard_available = [f for f in STANDARD_FEATURES if f in available_features]
    
    transformers = []
    
    if robust_available:
        robust_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median', add_indicator=False)),
            ('scaler', RobustScaler())
        ])
        transformers.append(('robust', robust_pipeline, robust_available))
    
    if standard_available:
        standard_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median', add_indicator=False)),
            ('scaler', StandardScaler())
        ])
        transformers.append(('standard', standard_pipeline, standard_available))
    
    preprocessor = ColumnTransformer(transformers, remainder='drop')
    
    return preprocessor


def build_full_pipeline(model_type: str = 'xgboost', 
                        feature_selection: bool = False,
                        available_features: list = None) -> Pipeline:
    """Build full modelling pipeline."""
    if available_features is None:
        available_features = NUMERIC_FEATURES
    
    preprocessor = build_preprocessing_pipeline(available_features)
    
    if model_type == 'xgboost':
        model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )
    elif model_type == 'logistic':
        model = LogisticRegression(
            C=0.1,
            solver='saga',
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    steps = [('preprocessor', preprocessor)]
    
    if feature_selection:
        selector = SelectKBest(score_func=f_classif, k=min(15, len(available_features)))
        steps.append(('selector', selector))
    
    steps.append(('model', model))
    
    return Pipeline(steps)
