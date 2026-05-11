"""
Model Evaluation & Diagnostics.

Generates 4-panel evaluation plots and saves metrics.
"""

import sys
sys.path.insert(0, '/Users/Rohan/Documents/Data/project_01_sme_credit_risk')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.calibration import calibration_curve
import joblib
import os

from src.models.evaluation_metrics import compute_credit_risk_metrics
from src.models.pipeline_builder import get_available_features

print("Loading data and models...")
df = pd.read_csv('data/processed/train_test_split/features_with_labels.csv')
feature_cols = get_available_features(df)
df['application_date'] = '2022-01-01'
X = df[feature_cols + ['application_date']].copy()
y = df['defaulted'].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

gb_pipeline = joblib.load('models/sme_credit_pipeline_xgb.joblib')
lr_pipeline = joblib.load('models/sme_credit_pipeline_lr.joblib')

y_prob_gb = gb_pipeline.predict_proba(X_test)[:, 1]
y_prob_lr = lr_pipeline.predict_proba(X_test)[:, 1]

# ============================================================
# 4-Panel Evaluation Figure
# ============================================================
print("Generating 4-panel evaluation plots...")
os.makedirs('notebooks/figures', exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. ROC Curve
ax = axes[0, 0]
for name, y_prob in [('GradientBoosting', y_prob_gb), ('LogisticRegression', y_prob_lr)]:
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = np.trapezoid(tpr, fpr)
    ax.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {auc:.3f})')
ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)

# 2. Precision-Recall Curve
ax = axes[0, 1]
for name, y_prob in [('GradientBoosting', y_prob_gb), ('LogisticRegression', y_prob_lr)]:
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    ax.plot(recall, precision, lw=2, label=name)
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curve')
ax.axhline(y_test.mean(), color='red', linestyle='--', label=f'Baseline ({y_test.mean():.3f})')
ax.legend()
ax.grid(True, alpha=0.3)

# 3. Calibration Plot
ax = axes[1, 0]
for name, y_prob in [('GradientBoosting', y_prob_gb), ('LogisticRegression', y_prob_lr)]:
    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
    ax.plot(prob_pred, prob_true, 's-', label=name)
ax.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
ax.set_xlabel('Mean Predicted Probability')
ax.set_ylabel('Fraction of Positives')
ax.set_title('Calibration Plot')
ax.legend()
ax.grid(True, alpha=0.3)

# 4. Score Distribution (KS Plot)
ax = axes[1, 1]
for name, y_prob, color in [('GradientBoosting', y_prob_gb, 'blue'), ('LogisticRegression', y_prob_lr, 'green')]:
    pos = y_prob[y_test == 1]
    neg = y_prob[y_test == 0]
    ax.hist(pos, bins=30, alpha=0.5, label=f'{name} - Default', color='red', density=True)
    ax.hist(neg, bins=30, alpha=0.5, label=f'{name} - Non-default', color=color, density=True)
ax.set_xlabel('Predicted Probability')
ax.set_ylabel('Density')
ax.set_title('Score Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('notebooks/figures/04_model_evaluation.png', dpi=150, bbox_inches='tight')
print("Saved: notebooks/figures/04_model_evaluation.png")
plt.close()

# ============================================================
# Feature Importance
# ============================================================
print("Generating feature importance plot...")
fig, ax = plt.subplots(figsize=(10, 8))

# GB feature importance
importance = gb_pipeline.named_steps['model'].feature_importances_
feature_names = feature_cols
imp_df = pd.DataFrame({'feature': feature_names, 'importance': importance})
imp_df = imp_df.sort_values('importance', ascending=True).tail(15)
imp_df.plot(x='feature', y='importance', kind='barh', ax=ax, color='steelblue')
ax.set_title('Top 15 Feature Importances (GradientBoosting)')
ax.set_xlabel('Importance')

plt.tight_layout()
plt.savefig('notebooks/figures/05_feature_importance.png', dpi=150, bbox_inches='tight')
print("Saved: notebooks/figures/05_feature_importance.png")
plt.close()

print("\nEvaluation complete!")
