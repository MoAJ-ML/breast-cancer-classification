# WDBC Classification Exercise

This educational project classifies the labels in the [UCI Breast Cancer Wisconsin (Diagnostic) dataset](https://archive.ics.uci.edu/dataset/17/breast-cancer-wisconsin-diagnostic), using its 569 rows and 30 features. UCI lists the dataset under CC BY 4.0. The model is **not a medical device or clinically validated diagnostic tool**; its small historical sample does not establish performance for current patients or other sites.

## Reproduce

```bash
pip install -r requirements.txt
python validate_model.py
python -m unittest discover -s tests
```

`validate_model.py` maps `M` to positive class 1 and `B` to 0, checks unique IDs, and makes stratified 60% training, 20% validation, and 20% untouched test partitions. Scaling occurs inside the training-only logistic-regression pipeline. Validation F1 chooses between logistic regression and random forest. The selected model is evaluated once on the untouched test with accuracy, malignant-class precision, recall, F1 and ROC-AUC. Results are written to `reports/validation_results.json`; the generated model is ignored by Git. Only load model files from trusted sources.

The notebook has been changed to select on validation and evaluate a separate final holdout. It also contains exploratory plots and optional SHAP/XGBoost steps; notebook outputs are not evidence of an independent clinical validation. The committed `visualizations/` files are historical until regenerated with the revised method.
Notebook-only packages such as Jupyter, SHAP and XGBoost are optional and are not needed for the validated script.

The 2026-11-10 script rerun selected logistic regression on validation F1 0.966; its untouched 114-row test had malignant-class precision 0.976, recall 0.952, F1 0.964, and ROC-AUC 0.995. These are dataset holdout scores, not clinical performance estimates.

No software license file is included; the dataset license does not automatically license this code. Suggested repository description: “Educational WDBC classifier with validation-selected models and a held-out test; nonclinical.”
