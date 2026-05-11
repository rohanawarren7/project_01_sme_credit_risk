What I'd do differently in v2:

1. **Graph-based entity resolution** — Current fuzzy matching + clustering works for 96% volume coverage, but graph embeddings from supplier-buyer networks would push accuracy to 90%+.

2. **Merchant embeddings** — Train item2vec on transaction baskets to learn merchant vectors. Similar merchants cluster in latent space, making resolution more robust to unseen variants.

3. **Online learning** — Current model is batch-trained. In production, I'd use river or scikit-multiflow for incremental updates as new defaults mature.

4. **Real Open Banking APIs** — Synthetic data proves the pipeline. Next step: integrate with Truelayer/Plaid sandbox for live transaction feeds.

5. **SHAP + regulatory reporting** — Build automated model risk documentation (SR 11-7 style) with versioned SHAP explanations for every retrain.

The MVP taught me that 80% of the value is in data engineering, not model selection. Clean entity resolution + thoughtful feature engineering beats a fancier algorithm every time.

#MLOps #DataEngineering #CreditRisk #LessonsLearned
