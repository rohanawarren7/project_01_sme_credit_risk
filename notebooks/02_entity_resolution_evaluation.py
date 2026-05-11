"""
Entity Resolution Evaluation Script.

This script:
1. Loads synthetic transaction data
2. Builds canonical dictionary from frequent normalised names
3. Runs the 3-pass MerchantResolver
4. Evaluates against ground truth built from known merchant taxonomy
5. Outputs metrics and saves results
"""

import sys
sys.path.insert(0, '/Users/Rohan/Documents/Data/project_01_sme_credit_risk')

import pandas as pd
import numpy as np
from src.entity_resolution.merchant_resolver import MerchantResolver
from src.entity_resolution.evaluation import evaluate_resolution
from src.data_ingestion.synthetic_data_generator import MERCHANT_TEMPLATES

print("Loading transaction data...")
transactions = pd.read_csv('data/raw/open_banking/transactions.csv')
print(f"Total transactions: {len(transactions):,}")

name_counts = transactions['merchant_name'].value_counts()
print(f"Unique merchant names: {len(name_counts):,}")

print("\nBuilding data-driven canonical dictionary...")
# Build canonical from unique names (not all transactions) using their frequency
resolver = MerchantResolver()
unique_raw_names = name_counts.index.tolist()
unique_freq = name_counts.values

# Normalise unique names and weight by frequency
normalised_counts = {}
for name, freq in zip(unique_raw_names, unique_freq):
    norm = resolver.normalise(name)
    normalised_counts[norm] = normalised_counts.get(norm, 0) + freq

# Keep normalised names that appear frequently
sorted_canonical = sorted(normalised_counts.items(), key=lambda x: x[1], reverse=True)
canonical = [name for name, freq in sorted_canonical if freq >= 5000]
print(f"Canonical dictionary size: {len(canonical)}")
print("Top canonical entries:", canonical[:15])

# Re-initialise with canonical dictionary
resolver = MerchantResolver(canonical_dictionary=canonical)

print("\nGenerating ground truth evaluation set...")
# Build ground truth by mapping each raw name to its known merchant group
ground_truth_rows = []
for raw in unique_raw_names:
    norm = resolver.normalise(raw)
    
    # Find which merchant taxonomy this belongs to
    true_canonical = norm
    for merchant_key, variants in MERCHANT_TEMPLATES.items():
        if raw in variants:
            # Use the most common normalised form of the first variant as canonical
            true_canonical = resolver.normalise(variants[0])
            break
    
    ground_truth_rows.append({
        'raw_name': raw,
        'true_canonical': true_canonical
    })

ground_truth = pd.DataFrame(ground_truth_rows)
print(f"Ground truth samples: {len(ground_truth)}")

print("\nRunning entity resolution (3-pass pipeline)...")
resolved = resolver.resolve(unique_raw_names, fuzzy_threshold=70, cluster_eps=0.3)
print(f"Resolved {len(resolved)} unique merchant names")

print("\nEvaluating...")
metrics = evaluate_resolution(resolved, ground_truth, transactions_df=transactions)

print("\n" + "="*50)
print("ENTITY RESOLUTION RESULTS")
print("="*50)
print(f"Overall Accuracy:     {metrics['overall_accuracy']:.1%}")
print(f"Coverage (by count):  {metrics['coverage']:.1%}")
if metrics['volume_coverage'] is not None:
    print(f"Coverage (by volume): {metrics['volume_coverage']:.1%}")
print(f"Samples Evaluated:    {metrics['n_evaluated']}")
print("\nPer-Method Breakdown:")
for method, stats in metrics['method_accuracy'].items():
    print(f"  {method:12s}: accuracy={stats['accuracy']:.1%}, n={stats['count']}")

print("\nTop Confusion Samples:")
if len(metrics['confusion_samples']) > 0:
    print(metrics['confusion_samples'][['raw', 'normalised', 'canonical', 'true_canonical', 'method']].head(10).to_string(index=False))
else:
    print("No confusion samples!")

# Save resolved mapping
import os
os.makedirs('data/processed/entity_resolution', exist_ok=True)
resolved.to_csv('data/processed/entity_resolution/merchant_mapping.csv', index=False)
print("\nSaved merchant mapping to data/processed/entity_resolution/merchant_mapping.csv")

# Save metrics
import json
with open('data/processed/entity_resolution/evaluation_metrics.json', 'w') as f:
    json.dump({
        'overall_accuracy': metrics['overall_accuracy'],
        'coverage': metrics['coverage'],
        'volume_coverage': metrics['volume_coverage'],
        'method_accuracy': {k: v for k, v in metrics['method_accuracy'].items()}
    }, f, indent=2)
print("Saved metrics to data/processed/entity_resolution/evaluation_metrics.json")
