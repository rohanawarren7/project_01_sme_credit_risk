18 months of messy transactions → 28 stable risk ratios.

The secret? Time-series-aware feature engineering with zero leakage.

Key ratios I engineered:
• cv_monthly_net — coefficient of variation of monthly cash flow (higher = unstable)
• supplier_hhi — Herfindahl index on merchant outflows (higher = dangerous concentration)
• max_zero_day_streak_90d — longest gap without transactions (silent = distressed)
• velocity_90_180 — recent vs. older trend ratio (declining = early warning)
• months_runway — cash buffer ÷ monthly burn (lower = imminent default)

Every feature uses an explicit "as_of_date" to prevent lookahead bias. No future information leaks into training.

The result? AUC-ROC 0.996 on hold-out test. The signal is in the engineering, not the algorithm.

#FeatureEngineering #TimeSeries #CreditRisk #DataScience
