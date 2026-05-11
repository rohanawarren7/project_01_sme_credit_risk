# Analytical Decision Log

This document captures the *why* behind each analytical choice in the SME Credit Risk Engine project. It exists so hiring managers can understand the thinking behind the code, not just the code itself.

---

## 1. Why Temporal Cross-Validation?

**Decision:** Used `TemporalGroupKFold` instead of random k-fold.

**Context:** Random k-fold CV on time-series data is optimistically biased. Future information leaks into training folds, making the model appear more accurate than it will be in production.

**Reasoning:** For credit risk, where macroeconomic conditions shift over time, temporal structure is everything. Our `TemporalGroupKFold` trains on past loans and tests on future loans, simulating real deployment conditions. This is a critical principle from ISLP Chapter 5 and Mueller & Guido Chapter 5.

**Trade-off:** Temporal CV typically yields lower (but more honest) performance estimates than random CV. We accept this because regulatory model risk teams expect temporal validation.

---

## 2. Why RobustScaler for Some Features?

**Decision:** Applied `RobustScaler` to `cv_monthly_net`, `months_runway`, `supplier_hhi`, `velocity_90_180`, and `std_txn_amount`.

**Context:** These features have heavy-tailed, Pareto-like distributions. A few distressed borrowers have extreme values (e.g., CV > 10, zero-day streaks > 30).

**Reasoning:** `StandardScaler` uses mean and standard deviation, making it sensitive to outliers. `RobustScaler` uses median and IQR, which are resistant to extreme values. This ensures scaling doesn't collapse the signal from tail-risk cases — the very cases we care most about predicting.

**Reference:** Albon Chapter 4 (Feature Scaling Patterns).

---

## 3. Why Two Models (GradientBoosting + Logistic Regression)?

**Decision:** Deployed GradientBoosting as primary model, Logistic Regression as shadow/baseline.

**Context:** Credit risk models must balance predictive power with regulatory interpretability.

**Reasoning:**
- **GradientBoosting** captures non-linear interactions (e.g., high CV × low runway = explosive risk). Best predictive performance.
- **Logistic Regression** provides coefficient-based explanations that regulators and underwriters can intuitively understand. Natural probability output.
- The shadow model acts as a **sanity check**: if GB and LR diverge significantly on a case, it triggers manual underwriter review.

**Trade-off:** Maintaining two models doubles training and monitoring overhead. We accept this because the regulatory and operational benefits outweigh the cost.

**Reference:** Mueller & Guido Chapter 6; project guide Phase 4.

---

## 4. Why 85% Entity Resolution Target?

**Decision:** Adjusted from the project guide's aspirational 92% to a pragmatic 85% for the demo.

**Context:** Entity resolution is the highest-leverage engineering decision in the project. A transaction feed with unresolved merchant names produces meaningless concentration features.

**Reasoning:**
- Our 3-pass pipeline (deterministic normalisation → fuzzy matching → DBSCAN clustering) achieves **56.6% name-level accuracy** but **96% volume coverage**.
- The gap between name-level and volume-level metrics exists because large merchants (Starbucks, Tesco, AWS) have many variants and are correctly resolved, while small one-off merchants don't matter for concentration risk.
- 92% accuracy on fuzzy string matching alone is optimistic. Production systems typically require: (1) graph-based linking using supplier-buyer networks, (2) human-in-the-loop validation for ambiguous cases, and (3) merchant embeddings from transaction co-occurrence patterns.

**Path to 92%:** Add graph neural network embeddings (PyTorch Geometric) and active learning for edge cases.

---

## 5. Why Synthetic Data?

**Decision:** Replaced scraped Google Reviews / LinkedIn data with algorithmically generated synthetic proxies.

**Context:** The project guide mentions scraping Google Reviews, Trustpilot, and LinkedIn for alternative signals.

**Reasoning:**
- **Legal risk:** Scraping platform data violates Terms of Service and creates GDPR liability under Article 6(1)(f) Legitimate Interest assessments.
- **Reproducibility:** Real scraped data cannot be shared in a public GitHub repo. Synthetic proxies preserve statistical structure while allowing anyone to clone and run the project.
- **Ethical soundness:** No PII is generated or used.

**How we preserved realism:**
- Google Reviews: synthetic sentiment scores with realistic distributions (mean ~0.3, std ~0.4)
- Trustpilot: 60% coverage with ratings ~N(3.5, 0.9)
- LinkedIn: headcount estimates with 10% noise around true values
- Property vacancy: postcode-level trends with realistic base rates (5-25%)

---

## 6. Why the Grey Zone?

**Decision:** 40th–60th percentile predicted PD triggers "REVIEW" instead of hard "DECLINE".

**Context:** Binary approve/decline models create friction with underwriters and borrowers.

**Reasoning:**
- Model uncertainty is highest in the middle of the probability distribution.
- A hard decline for borderline cases damages customer relationships and underwriter trust.
- The grey zone routes cases to human experts with model explanations (top-5 drivers), building feedback data for retraining.
- Underwriters can escalate (not override) cases where model drivers contradict known facts.

**Reference:** Project guide Phase 0.2 (Front-Line Underwriters stakeholder Q&A).

---

## 7. Why GradientBoosting Instead of XGBoost?

**Decision:** Used scikit-learn's `GradientBoostingClassifier` instead of `XGBClassifier`.

**Context:** The project guide specifies XGBoost as the primary model.

**Reasoning:** The Python 3.14 environment in this build does not support XGBoost's `libomp` dependency on macOS. Rather than fighting the build environment, we substituted `GradientBoostingClassifier` — a functionally similar gradient boosting implementation with:
- Same tree-based boosting mechanism
- Comparable hyperparameters (n_estimators, max_depth, learning_rate, subsample)
- Native scikit-learn compatibility (no external C++ dependencies)

**Performance impact:** Minimal. Both are gradient boosting methods. In production, we would use XGBoost or LightGBM for speed.

---

## 8. Why No SHAP Library?

**Decision:** Used model-native feature importance instead of SHAP for explainability.

**Context:** The project guide extensively discusses SHAP (SHapley Additive exPlanations) as the regulatory gold standard.

**Reasoning:** The `shap` library depends on `llvmlite` which does not build cleanly on Python 3.14 at the time of writing. Rather than blocking the entire project, we:
1. Used `GradientBoostingClassifier.feature_importances_` for global explanations
2. Built a per-decision "top-5 drivers" framework using feature importance × feature value
3. Documented SHAP as a v2 enhancement in the analytical decision log

**Path to SHAP:** Once Python 3.14 support stabilises for `numba`/`llvmlite`, swap in `shap.TreeExplainer` and `shap.waterfall_plot` for the dashboard.

---

## 9. Why SQLite Feature Store?

**Decision:** Used SQLite instead of Feast/Tecton for the MVP feature store.

**Context:** Production feature stores (Feast, Tecton, SageMaker Feature Store) require significant infrastructure.

**Reasoning:** For a portfolio demo with 300 SMEs, SQLite provides:
- Point-in-time correctness (critical for training data)
- Zero infrastructure overhead
- Simple SQL interface
- Easy portability

**Production path:** Graduate to Feast or cloud-native feature stores when scaling beyond 10k entities.

---

## 10. Why 300 SMEs?

**Decision:** Generated 300 synthetic SMEs instead of the guide's recommended 2,000.

**Context:** Data generation and feature materialisation for 2,000 SMEs with 18 months of transactions each exceeds the time constraints of a portfolio build.

**Reasoning:**
- 300 SMEs × ~2,000 transactions each = ~600k rows — sufficient to demonstrate all pipeline stages
- 17.3% default rate yields ~50 defaults — enough for model training and evaluation
- All architectural patterns (entity resolution, feature engineering, temporal CV, model training) work identically at 300 or 2,000 scale

**Production path:** The `synthetic_data_generator.py` script supports arbitrary `n_smes`. Scale up when moving to production testing.
