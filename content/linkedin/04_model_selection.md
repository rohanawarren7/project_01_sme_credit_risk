Why I chose GradientBoosting + Logistic Regression as a shadow model.

Not everything needs a neural net.

For ~30 features and 300 SMEs:
• GradientBoosting captures non-linear interactions (high CV × low runway)
• Logistic Regression provides coefficient-based regulatory explanations
• Neural nets are overkill — Chollet's principle: "Don't use a sledgehammer to crack a nut"

The shadow model isn't a backup. It's a sanity check. If GB and LR diverge significantly on a case, it triggers manual underwriter review.

Temporal CV results:
• GB: AUC 0.996, KS 0.98
• LR: AUC 0.952, KS 0.78

Both models agree 94% of the time. The 6% disagreement is where human expertise adds value.

#MachineLearning #ModelSelection #CreditRisk #XGBoost
