"""
Feature Materialisation Script.

Computes and stores feature vectors for all SMEs as of their application date.
"""

import sys
sys.path.insert(0, '/Users/Rohan/Documents/Data/project_01_sme_credit_risk')

import pandas as pd
import numpy as np
from datetime import datetime
import json
import os

from src.entity_resolution.merchant_resolver import MerchantResolver
from src.features.transaction_engineer import TransactionFeatureEngineer
from src.features.alternative_engineer import AlternativeFeatureEngineer
from src.features.feature_store import FeatureStore

print("Loading data...")
transactions = pd.read_csv('data/raw/open_banking/transactions.csv')
defaults = pd.read_csv('data/raw/default_labels.csv')
sme_profiles = pd.read_csv('data/raw/sme_profiles.csv')

# Enrichment data
google = pd.read_csv('data/raw/google_reviews/google_reviews.csv')
trustpilot = pd.read_csv('data/raw/trustpilot/trustpilot.csv')
linkedin = pd.read_csv('data/raw/linkedin/linkedin.csv')
property = pd.read_csv('data/raw/property_vacancy/property_vacancy.csv')
companies_house = pd.read_csv('data/raw/companies_house/companies_house.csv')

# Entity resolution
print("Loading entity resolution mapping...")
merchant_mapping = pd.read_csv('data/processed/entity_resolution/merchant_mapping.csv')

# Initialise feature engineers
print("Initialising feature engineers...")
txn_engineer = TransactionFeatureEngineer(transactions, merchant_mapping_df=merchant_mapping)
alt_engineer = AlternativeFeatureEngineer(google, trustpilot, linkedin, property, companies_house)

# Feature store
store = FeatureStore(db_path='data/processed/feature_vectors.db')
# Clear existing
os.remove('data/processed/feature_vectors.db') if os.path.exists('data/processed/feature_vectors.db') else None
store = FeatureStore(db_path='data/processed/feature_vectors.db')

print("\nMaterialising features for all SMEs...")
as_of_date = pd.Timestamp('2023-06-30')

all_features = []
for idx, sme in sme_profiles.iterrows():
    if (idx + 1) % 50 == 0:
        print(f"  Processed {idx + 1}/{len(sme_profiles)} SMEs...")
    
    # Transaction features
    txn_features = txn_engineer.stability_ratios(sme['company_id'], as_of_date)
    
    # Alternative features
    alt_features = alt_engineer.build_features(
        sme['company_id'], as_of_date, postcode=sme['postcode']
    )
    
    # Merge
    features = {**txn_features, **alt_features}
    features['company_id'] = sme['company_id']
    features['feature_date'] = as_of_date.strftime('%Y-%m-%d')
    
    # Store
    store.materialise_features(sme['company_id'], as_of_date, features)
    all_features.append(features)

print(f"\nMaterialised features for {len(all_features)} SMEs")

# Save as CSV for convenience
features_df = pd.DataFrame(all_features)
features_df.to_csv('data/processed/feature_vectors/all_features.csv', index=False)
print("Saved to data/processed/feature_vectors/all_features.csv")

# Merge with default labels for training
labels = defaults[['company_id', 'defaulted']].copy()
features_df = features_df.merge(labels, on='company_id', how='left')
features_df.to_csv('data/processed/train_test_split/features_with_labels.csv', index=False)
print("Saved features_with_labels.csv")

print("\nFeature materialisation complete!")
