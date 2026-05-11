"Starbucks Ltd #084381", "Starbucks", "SBUK RETAIL" — three names, one merchant.

If you can't resolve these, your concentration risk features are meaningless.

My 3-pass pipeline:
1. Deterministic normalisation (strip legal suffixes, processor prefixes, store codes)
2. Fuzzy match against a canonical dictionary (rapidfuzz, token_sort_ratio ≥ 70)
3. DBSCAN clustering on TF-IDF character n-grams for the leftovers

Current accuracy: 56.6% on name-level, 96% volume coverage.

The trick? Weight coverage by GBP volume, not transaction count. A missed Starbucks matters more than a missed corner shop.

Code + evaluation notebook in the repo.

#NLP #EntityResolution #DataEngineering #CreditRisk
