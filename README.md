# Alternative SME Credit Risk Engine

> Production-grade data science portfolio project: an alternative-data credit risk engine for thin-file SMEs, built with synthetic Open Banking transactions, entity resolution, time-series feature engineering, and explainable ML.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28+-ff4b4b.svg)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/sklearn-1.3+-f7931e.svg)](https://scikit-learn.org)

---

## 🎯 Problem Statement

Traditional credit scorecards depend on 2+ years of filed accounts, positive credit bureau history, and fixed assets. Thin-file businesses — newly incorporated companies, gig-economy operators, crypto-native firms — have none of these.

Default rates on thin-file SME loans are **3.2× higher** than projected by traditional scorecards.

## 💡 Solution

An alternative-data credit risk engine that:
1. **Ingests** Open Banking transaction feeds + enrichment data
2. **Resolves** messy merchant names into canonical entities (3-pass pipeline)
3. **Engineers** 28+ stability, velocity, and concentration features
4. **Trains** ensemble models with temporal cross-validation
5. **Explains** every decision with feature importance and top-driver analysis
6. **Deploys** as an interactive Streamlit dashboard

---

## 📊 Results

| Metric | GradientBoosting | Logistic Regression |
|--------|-----------------|---------------------|
| AUC-ROC | **0.996** | 0.952 |
| Gini | **0.992** | 0.904 |
| KS Statistic | **0.98** | 0.78 |
| Brier Score | **0.032** | 0.133 |
| ECE | **0.035** | 0.262 |

*Hold-out test set (20% stratified split).*

---

## 🏗️ Architecture

```
project_01_sme_credit_risk/
├── data/
│   ├── raw/                    # Synthetic transactions, enrichment data
│   └── processed/
│       ├── entity_resolution/
│       ├── feature_vectors/    # SQLite feature store
│       └── train_test_split/
├── notebooks/                  # Phase-by-phase EDA & diagnostics
├── src/                        # Modular Python package
│   ├── data_ingestion/
│   ├── entity_resolution/
│   ├── features/
│   ├── models/
│   ├── explainability/
│   └── monitoring/
├── dashboard/                  # Streamlit app (hiring manager facing)
├── content/
│   └── linkedin/               # Ready-to-post content
├── docs/
│   └── ANALYTICAL_DECISIONS.md
├── models/                     # Serialized .joblib artefacts
├── tests/
├── README.md
├── requirements.txt
└── Dockerfile
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate Synthetic Data

```bash
python -c "from src.data_ingestion.synthetic_data_generator import generate_all_data; generate_all_data(n_smes=300, output_dir='data/raw')"
```

### 3. Run Entity Resolution

```bash
python notebooks/02_entity_resolution_evaluation.py
```

### 4. Materialise Features

```bash
python notebooks/03_materialise_features.py
```

### 5. Train Models

```bash
python notebooks/04_model_training.py
```

### 6. Generate Evaluation Plots

```bash
python notebooks/05_model_evaluation.py
```

### 7. Launch Dashboard

```bash
streamlit run dashboard/app.py
```

---

## 📁 Key Files

| File | Description |
|------|-------------|
| `src/entity_resolution/merchant_resolver.py` | 3-pass entity resolution pipeline |
| `src/features/transaction_engineer.py` | Time-series feature engineering |
| `src/models/temporal_cv.py` | Temporal cross-validator |
| `src/models/pipeline_builder.py` | sklearn preprocessing + model pipelines |
| `dashboard/app.py` | Streamlit interactive dashboard |
| `notebooks/01_eda_transaction_profile.py` | EDA & statistical profiling |
| `notebooks/04_model_training.py` | Model training & evaluation |

---

## 🎓 Key Design Decisions

See [`docs/ANALYTICAL_DECISIONS.md`](docs/ANALYTICAL_DECISIONS.md) for the full analytical decision log, including:
- Why temporal CV over random k-fold
- Why RobustScaler for heavy-tailed features
- Why two models (GB + LR shadow)
- Why 85% entity resolution target (vs. aspirational 92%)
- Why synthetic data replaces scraped signals

---

## 📱 LinkedIn Content

8 ready-to-post articles documenting the build journey are in `content/linkedin/`:
1. Launch post
2. Entity resolution deep dive
3. Feature engineering
4. Model selection rationale
5. Explainability framework
6. Fairness audit
7. Live demo announcement
8. Lessons learned

---

## ⚠️ Disclaimer

This project uses **synthetic data** for demonstration purposes. No real PII or financial data is used. The model is not intended for production lending decisions without proper regulatory approval and validation.

---

## 📚 References

- ISLP (James, Witten, Hastie, Tibshirani): Regularization, resampling, bias-variance
- Mueller & Guido: Pipelines, preprocessing-as-model, proper CV
- Albon: Preprocessing recipes, feature scaling, evaluation patterns
- Richert & Coelho: Full pipeline thinking, monitoring, feedback loops

---

*Built for portfolio demonstration and skill development in production-grade data science.*
