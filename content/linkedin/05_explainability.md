Every automated decline needs a human-readable explanation.

In credit risk, "the model said no" is not an acceptable answer to a regulator — or a borrower.

My explainability framework:
1. **Global feature importance** — which variables drive the model?
2. **Top-5 drivers per decision** — why did THIS company get declined?
3. **Grey-zone logic** — 40th-60th percentile PD routes to human review, not hard decline

Top features in my model:
• supplier_hhi (concentration risk)
• cv_monthly_net (cash flow volatility)
• total_outflow (burn rate proxy)
• active_days_ratio (business activity)
• months_runway (liquidity buffer)

The goal isn't perfect automation. It's augmented underwriting: the model handles the obvious cases, humans handle the edge cases.

#ExplainableAI #SHAP #CreditRisk #ResponsibleAI
