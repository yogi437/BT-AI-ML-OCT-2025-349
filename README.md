Titanic ML Project — Step-by-Step Workflow
------------------------------------------
Files:
  1_data_loading.py          → Load and inspect raw data
  2_data_cleaning.py         → Handle missing values & feature engineering
  3_eda.py                   → Create EDA visualizations
  4_model_logistic_regression.py  → Train & evaluate Logistic Regression
  5_model_decision_tree.py   → Train & evaluate Decision Tree
  6_submission.py            → Train on full data and create submission CSVs

Run order:
  1 → 2 → 3 → 4 → 5 → 6

Outputs:
  - train_clean.csv, test_clean.csv
  - submission_logreg.csv, submission_dtree.csv
