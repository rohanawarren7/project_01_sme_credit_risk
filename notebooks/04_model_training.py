"""
Model Training Script.

Trains GradientBoosting and Logistic Regression models with temporal CV.
"""

import sys
sys.path.insert(0, '/Users/Rohan/Documents/Data/project_01_sme_credit_risk')

import pandas as pd
import numpy as np
import json
import joblib
import os
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss, make_scorer

from src.models.pipeline_builder import build_full_pipeline, get_available_features
from src.models.temporal_cv import TemporalGroupKFold
from src.models.evaluation_metrics import compute_credit_risk_metrics

print("Loading features...")
df = pd.read_csv('data/processed/train_test_split/features_with_labels.csv')

# Drop non-feature columns
feature_cols = get_available_features(df)
print(f"Available features: {len(feature_cols)}")
print(f"Features: {feature_cols}")

df['application_date'] = '2022-01-01'
X = df[feature_cols + ['application_date']].copy()
y = df['defaulted'].copy()

print(f"Samples: {len(X)}")
print(f"Default rate: {y.mean():.1%}")

# Custom scorer for KS that works with sklearn
def ks_scorer(estimator, X, y):
    y_prob = estimator.predict_proba(X)[:, 1]
    from src.models.evaluation_metrics import ks_statistic
    return ks_statistic(y, y_prob)

# Define scoring
scoring = {
    'roc_auc': 'roc_auc',
    'brier': make_scorer(brier_score_loss, needs_proba=True, greater_is_better=False),
    'ks': make_scorer(ks_scorer, needs_proba=False),
}

# Temporal CV
cv = TemporalGroupKFold(n_splits=5, test_size=0.2, date_col='application_date')

# ============================================================
# Train GradientBoosting
# ============================================================
print("\n" + "="*50)
print("Training GradientBoosting...")
print("="*50)

gb_pipeline = build_full_pipeline(model_type='xgboost', available_features=feature_cols)
gb_results = cross_validate(
    gb_pipeline, X, y,
    cv=cv,
    scoring=scoring,
    return_train_score=True,
    n_jobs=1,
    return_estimator=True
)

print(f"Mean ROC-AUC: {np.nanmean(gb_results['test_roc_auc']):.4f}")
print(f"Mean KS:      {np.nanmean(gb_results['test_ks']):.4f}")
print(f"Mean Brier:   {-np.nanmean(gb_results['test_brier']):.4f}")

# ============================================================
# Train Logistic Regression
# ============================================================
print("\n" + "="*50)
print("Training Logistic Regression...")
print("="*50)

lr_pipeline = build_full_pipeline(model_type='logistic', available_features=feature_cols)
lr_results = cross_validate(
    lr_pipeline, X, y,
    cv=cv,
    scoring=scoring,
    return_train_score=True,
    n_jobs=1,
    return_estimator=True
)

print(f"Mean ROC-AUC: {np.nanmean(lr_results['test_roc_auc']):.4f}")
print(f"Mean KS:      {np.nanmean(lr_results['test_ks']):.4f}")
print(f"Mean Brier:   {-np.nanmean(lr_results['test_brier']):.4f}")

# ============================================================
# Hold-out evaluation
# ============================================================
print("\n" + "="*50)
print("Hold-out Evaluation")
print("="*50)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

gb_pipeline.fit(X_train, y_train)
y_prob_gb = gb_pipeline.predict_proba(X_test)[:, 1]

lr_pipeline.fit(X_train, y_train)
y_prob_lr = lr_pipeline.predict_proba(X_test)[:, 1]

metrics_gb = compute_credit_risk_metrics(y_test.values, y_prob_gb)
metrics_lr = compute_credit_risk_metrics(y_test.values, y_prob_lr)

print("\nGradientBoosting Hold-out:")
print(f"  AUC-ROC:      {metrics_gb['auc_roc']:.4f}")
print(f"  Gini:         {metrics_gb['gini']:.4f}")
print(f"  KS:           {metrics_gb['ks_statistic']:.4f}")
print(f"  Brier:        {metrics_gb['brier_score']:.4f}")
print(f"  ECE:          {metrics_gb['ece']:.4f}")

print("\nLogistic Regression Hold-out:")
print(f"  AUC-ROC:      {metrics_lr['auc_roc']:.4f}")
print(f"  Gini:         {metrics_lr['gini']:.4f}")
print(f"  KS:           {metrics_lr['ks_statistic']:.4f}")
print(f"  Brier:        {metrics_lr['brier_score']:.4f}")
print(f"  ECE:          {metrics_lr['ece']:.4f}")

# ============================================================
# Save models and metrics
# ============================================================
print("\nSaving models and metrics...")
os.makedirs('models', exist_ok=True)

# Retrain on all data
gb_pipeline.fit(X, y)
lr_pipeline.fit(X, y)

joblib.dump(gb_pipeline, 'models/sme_credit_pipeline_xgb.joblib')
joblib.dump(lr_pipeline, 'models/sme_credit_pipeline_lr.joblib')
joblib.dump(feature_cols, 'models/feature_names.joblib')

# Save metrics
metrics = {
    'gradientboosting': {
        'roc_auc': float(metrics_gb['auc_roc']),
        'gini': float(metrics_gb['gini']),
        'ks': float(metrics_gb['ks_statistic']),
        'brier': float(metrics_gb['brier_score']),
        'ece': float(metrics_gb['ece']),
    },
    'logistic': {
        'roc_auc': float(metrics_lr['auc_roc']),
        'gini': float(metrics_lr['gini']),
        'ks': float(metrics_lr['ks_statistic']),
        'brier': float(metrics_lr['brier_score']),
        'ece': float(metrics_lr['ece']),
    }
}

with open('models/holdout_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print("\nModel training complete!")
print("Saved: models/sme_credit_pipeline_xgb.joblib")
print("Saved: models/sme_credit_pipeline_lr.joblib")
print("Saved: models/feature_names.joblib")
print("Saved: models/holdout_metrics.json")
