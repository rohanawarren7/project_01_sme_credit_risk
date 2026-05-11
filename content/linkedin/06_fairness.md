I audited my own credit risk model for proxy discrimination.

The risk: a feature like "merchant_postcode_entropy" could correlate with ethnicity, becoming a proxy for protected characteristics.

My 3-gate fairness framework:
1. **Pre-model**: correlation scan between features and demographic proxies
2. **In-model**: demographic parity analysis on available protected attributes
3. **Post-model**: flag if any top-3 driver is a known proxy

Finding: the model's top drivers are financial (supplier concentration, cash flow volatility, burn rate) — not demographic. No proxy discrimination detected in this synthetic dataset.

Fairness isn't a checkbox. It's a continuous audit.

#ResponsibleAI #Fairness #MachineLearning #CreditRisk #EthicsInAI
