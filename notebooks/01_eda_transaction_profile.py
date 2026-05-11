"""
EDA & Statistical Profiling for SME Credit Risk.

This script:
1. Loads synthetic transactions and default labels
2. Computes per-entity statistical summaries
3. Generates trajectory plots for defaulters vs survivors
4. Validates distributional assumptions
"""

import sys
sys.path.insert(0, '/Users/Rohan/Documents/Data/project_01_sme_credit_risk')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

print("Loading data...")
transactions = pd.read_csv('data/raw/open_banking/transactions.csv', parse_dates=['transaction_date'])
defaults = pd.read_csv('data/raw/default_labels.csv')
sme_profiles = pd.read_csv('data/raw/sme_profiles.csv')

print(f"Transactions: {len(transactions):,}")
print(f"SMEs: {len(sme_profiles):,}")
print(f"Default rate: {defaults['defaulted'].mean():.1%}")

# Merge default info
transactions = transactions.merge(defaults[['company_id', 'defaulted', 'default_date']], on='company_id', how='left')

# ============================================================
# 1. Per-Entity Statistical Profiles
# ============================================================
print("\nComputing per-entity statistical profiles...")

profiles = []
for company_id, group in transactions.groupby('company_id'):
    group = group.sort_values('transaction_date')
    amounts = group['amount']
    
    # Inflows and outflows
    inflows = group[group['amount'] > 0]['amount']
    outflows = group[group['amount'] < 0]['amount']
    
    # Daily aggregates
    daily = group.groupby('transaction_date')['amount'].sum()
    daily_in = group[group['amount'] > 0].groupby('transaction_date')['amount'].sum()
    daily_out = group[group['amount'] < 0].groupby('transaction_date')['amount'].sum().abs()
    
    profile = {
        'company_id': company_id,
        'defaulted': group['defaulted'].iloc[0],
        'n_transactions': len(group),
        'total_volume': amounts.sum(),
        'total_inflow': inflows.sum() if len(inflows) > 0 else 0,
        'total_outflow': outflows.sum() if len(outflows) > 0 else 0,
        'mean_txn': amounts.mean(),
        'median_txn': amounts.median(),
        'std_txn': amounts.std(),
        'cv_txn': amounts.std() / abs(amounts.mean()) if amounts.mean() != 0 else np.nan,
        'skewness': stats.skew(amounts.dropna()),
        'kurtosis': stats.kurtosis(amounts.dropna()),
        'max_daily_volume': daily.max() if len(daily) > 0 else 0,
        'min_daily_volume': daily.min() if len(daily) > 0 else 0,
        'active_days': group['transaction_date'].nunique(),
        'active_days_ratio': group['transaction_date'].nunique() / (group['transaction_date'].max() - group['transaction_date'].min()).days,
        'top_5_concentration': amounts.abs().nlargest(5).sum() / amounts.abs().sum() if amounts.abs().sum() > 0 else 0,
        'days_since_last_txn': (transactions['transaction_date'].max() - group['transaction_date'].max()).days,
        'mean_daily_inflow': daily_in.mean() if len(daily_in) > 0 else 0,
        'mean_daily_outflow': daily_out.mean() if len(daily_out) > 0 else 0,
        'cv_daily_net': daily.std() / abs(daily.mean()) if daily.mean() != 0 else np.nan,
    }
    profiles.append(profile)

profiles_df = pd.DataFrame(profiles)
print(f"Computed profiles for {len(profiles_df)} entities")

# Save profiles
os.makedirs('data/processed', exist_ok=True)
profiles_df.to_csv('data/processed/entity_profiles.csv', index=False)

# ============================================================
# 2. Distributional Validation
# ============================================================
print("\nValidating distributional assumptions...")

defaulted = profiles_df[profiles_df['defaulted'] == 1]
survived = profiles_df[profiles_df['defaulted'] == 0]

validation = {
    'Transaction amounts right-skewed': stats.skew(transactions['amount'].dropna()) > 0,
    'CV bimodal (defaulters vs survivors)': defaulted['cv_txn'].median() > survived['cv_txn'].median(),
    'Defaulters have higher CV': defaulted['cv_txn'].median() > survived['cv_txn'].median(),
    'Top-5 concentration Pareto-like': profiles_df['top_5_concentration'].median() > 0.3,
    'Defaulters higher concentration': defaulted['top_5_concentration'].median() > survived['top_5_concentration'].median(),
    'Active days ratio lower for defaulters': defaulted['active_days_ratio'].median() < survived['active_days_ratio'].median(),
}

print("\nValidation Results:")
for check, result in validation.items():
    print(f"  {'✓' if result else '✗'} {check}: {result}")

# ============================================================
# 3. Visualization Suite
# ============================================================
print("\nGenerating visualizations...")
os.makedirs('notebooks/figures', exist_ok=True)

# Figure 1: Distribution comparison
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# CV comparison
ax = axes[0, 0]
ax.hist(survived['cv_txn'].dropna(), bins=30, alpha=0.6, label='Survived', density=True, color='green')
ax.hist(defaulted['cv_txn'].dropna(), bins=30, alpha=0.6, label='Defaulted', density=True, color='red')
ax.set_xlabel('Coefficient of Variation (CV)')
ax.set_ylabel('Density')
ax.set_title('Transaction CV: Defaulters vs Survivors')
ax.legend()
ax.set_xlim(0, 5)

# Top-5 concentration
ax = axes[0, 1]
ax.hist(survived['top_5_concentration'].dropna(), bins=30, alpha=0.6, label='Survived', density=True, color='green')
ax.hist(defaulted['top_5_concentration'].dropna(), bins=30, alpha=0.6, label='Defaulted', density=True, color='red')
ax.set_xlabel('Top-5 Transaction Concentration')
ax.set_ylabel('Density')
ax.set_title('Concentration Risk: Defaulters vs Survivors')
ax.legend()

# Active days ratio
ax = axes[0, 2]
ax.hist(survived['active_days_ratio'].dropna(), bins=30, alpha=0.6, label='Survived', density=True, color='green')
ax.hist(defaulted['active_days_ratio'].dropna(), bins=30, alpha=0.6, label='Defaulted', density=True, color='red')
ax.set_xlabel('Active Days Ratio')
ax.set_ylabel('Density')
ax.set_title('Activity Level: Defaulters vs Survivors')
ax.legend()

# Transaction amount distribution
ax = axes[1, 0]
sample_txns = transactions.sample(n=min(50000, len(transactions)), random_state=42)
ax.hist(sample_txns[sample_txns['amount'] > 0]['amount'], bins=50, alpha=0.6, label='Inflows', color='blue')
ax.hist(sample_txns[sample_txns['amount'] < 0]['amount'].abs(), bins=50, alpha=0.6, label='Outflows', color='orange')
ax.set_xlabel('Amount (£)')
ax.set_ylabel('Frequency')
ax.set_title('Transaction Amount Distribution (Sample)')
ax.set_xlim(0, 5000)
ax.legend()

# Daily net position distribution
ax = axes[1, 1]
daily_net = transactions.groupby(['company_id', 'transaction_date'])['amount'].sum().reset_index()
daily_net = daily_net.merge(defaults[['company_id', 'defaulted']], on='company_id', how='left')
ax.hist(daily_net[daily_net['defaulted'] == 0]['amount'], bins=50, alpha=0.6, label='Survived', density=True, color='green')
ax.hist(daily_net[daily_net['defaulted'] == 1]['amount'], bins=50, alpha=0.6, label='Defaulted', density=True, color='red')
ax.set_xlabel('Daily Net Position (£)')
ax.set_ylabel('Density')
ax.set_title('Daily Net Position Distribution')
ax.set_xlim(-5000, 5000)
ax.legend()

# Industry default rates
ax = axes[1, 2]
industry_defaults = sme_profiles.merge(defaults[['company_id', 'defaulted']], on='company_id', how='left')
industry_rates = industry_defaults.groupby('industry')['defaulted'].mean().sort_values(ascending=False)
industry_rates.plot(kind='bar', ax=ax, color='steelblue')
ax.set_xlabel('Industry')
ax.set_ylabel('Default Rate')
ax.set_title('Default Rate by Industry')
ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('notebooks/figures/01_eda_distributions.png', dpi=150, bbox_inches='tight')
print("Saved: notebooks/figures/01_eda_distributions.png")
plt.close()

# Figure 2: Time-series trajectories
print("Generating trajectory plots...")

fig, axes = plt.subplots(3, 2, figsize=(16, 14))
axes = axes.flatten()

# Sample defaulters, survivors, and grey-zone
sample_defaults = defaulted.sample(min(2, len(defaulted)), random_state=42)['company_id'].tolist()
sample_survived = survived.sample(min(2, len(survived)), random_state=42)['company_id'].tolist()

sample_ids = sample_defaults + sample_survived
labels = ['Default'] * len(sample_defaults) + ['Survive'] * len(sample_survived)
colors = {'Default': '#e74c3c', 'Survive': '#2ecc71'}

for idx, (company_id, label) in enumerate(zip(sample_ids, labels)):
    ax = axes[idx]
    entity_df = transactions[transactions['company_id'] == company_id].copy()
    entity_df = entity_df.set_index('transaction_date').sort_index()
    
    daily = entity_df.groupby(entity_df.index)['amount'].sum()
    rolling = daily.rolling('30D').mean()
    
    ax.plot(daily.index, daily.values, alpha=0.3, color='grey', label='Daily net')
    ax.plot(rolling.index, rolling.values, color=colors[label], linewidth=2, label='30d rolling mean')
    ax.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax.set_title(f'{company_id} — {label}')
    ax.legend(loc='upper left')
    ax.set_ylabel('Net Position (£)')

# Hide unused subplots
for idx in range(len(sample_ids), 6):
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig('notebooks/figures/02_eda_trajectories.png', dpi=150, bbox_inches='tight')
print("Saved: notebooks/figures/02_eda_trajectories.png")
plt.close()

# Figure 3: Correlation heatmap of key features
print("Generating correlation heatmap...")
fig, ax = plt.subplots(figsize=(12, 10))

corr_features = ['cv_txn', 'top_5_concentration', 'active_days_ratio', 
                 'mean_txn', 'std_txn', 'days_since_last_txn',
                 'mean_daily_inflow', 'mean_daily_outflow', 'cv_daily_net',
                 'defaulted']

corr_matrix = profiles_df[corr_features].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', 
            center=0, vmin=-1, vmax=1, ax=ax, square=True)
ax.set_title('Feature Correlation Matrix')

plt.tight_layout()
plt.savefig('notebooks/figures/03_eda_correlation.png', dpi=150, bbox_inches='tight')
print("Saved: notebooks/figures/03_eda_correlation.png")
plt.close()

print("\nEDA complete!")
print(f"Profiles saved: data/processed/entity_profiles.csv")
print(f"Figures saved: notebooks/figures/")
