# 5_prediction_submission.py
"""Load best_model.joblib (created by 3_model_training.py), predict on test set,
convert back from log scale and write submission.csv matching sample format.
Usage: python 5_prediction_submission.py
Requires: pandas, numpy, joblib
"""
import pandas as pd
import numpy as np
import joblib
import os

MODEL_FILE = "best_model.joblib"
DATA_PRE = "data_preprocessed.csv"
OUT_SUB = "submission.csv"

def main():
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"{MODEL_FILE} not found. Run 3_model_training.py first.")
    model = joblib.load(MODEL_FILE)
    df = pd.read_csv(DATA_PRE)
    test = df[df['is_train']==0].copy()
    ids = test['Id'] if 'Id' in test.columns else pd.Series(range(len(test)))
    X_test = test.drop(columns=['SalePrice','is_train','Id'], errors='ignore')
    preds_log = model.predict(X_test)
    preds = np.expm1(preds_log)
    submission = pd.DataFrame({'Id': ids, 'SalePrice': preds})
    submission.to_csv(OUT_SUB, index=False)
    print('Saved', OUT_SUB)

if __name__ == '__main__':
    main()
