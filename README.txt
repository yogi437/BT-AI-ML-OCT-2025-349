House Price Regression - Code-only package
----------------------------------------
This zip contains only Python code files. Download the Kaggle dataset and place the following files
in the same folder before running the scripts:

- train.csv
- test.csv
- sample_submission.csv
- data_description.txt  (optional but recommended)

Run order (recommended):

1) python 1_data_cleaning.py
   - Produces: data_merged.csv and EDA plots in eda_plots/
2) python 2_feature_engineering.py
   - Produces: data_preprocessed.csv, encoding_info.txt, and fe_plots/
3) python 3_model_training.py
   - Trains LinearRegression, RandomForest, XGBoost. Saves best_model.joblib
4) python 4_cross_validation.py
   - (Optional) Generates out-of-fold predictions oof_predictions.csv
5) python 5_prediction_submission.py
   - Produces submission.csv ready for Kaggle upload

Requirements:
pandas, numpy, scikit-learn, xgboost, joblib, matplotlib

Notes:
- The scripts use simple, readable defaults. Feel free to expand feature engineering,
  try target encoding, hyperparameter tuning (RandomizedSearchCV / Optuna), and ensembling.
- When generating plots, matplotlib is used (no seaborn). Each plot is saved to disk.
