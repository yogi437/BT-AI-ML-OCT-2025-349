# 4_cross_validation.py
"""Standalone cross-validation helper that trains a specified model and returns per-fold predictions.
Useful for stacking or out-of-fold predictions.
Usage: python 4_cross_validation.py
Requires: pandas, scikit-learn, joblib, numpy
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from math import sqrt

DATA_PRE = "data_preprocessed.csv"

def rmse(y_true, y_pred):
    return sqrt(((y_true - y_pred)**2).mean())

def oof_predictions(model, X, y, n_splits=5, random_state=42):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_preds = np.zeros(len(X))
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)
        oof_preds[val_idx] = preds
        print(f"Fold {fold+1} RMSE: {rmse(y_val, preds):.5f}")
    print("Overall OOF RMSE:", rmse(y, oof_preds))
    return oof_preds

def main():
    df = pd.read_csv(DATA_PRE)
    train = df[df['is_train']==1].copy()
    X = train.drop(columns=['SalePrice','is_train','Id'], errors='ignore')
    y = train['SalePrice']
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    oof = oof_predictions(model, X, y)
    # Save oof to csv
    out = train[['Id']].copy() if 'Id' in train.columns else pd.DataFrame({'Id': range(len(train))})
    out['oof_pred'] = oof
    out.to_csv('oof_predictions.csv', index=False)
    print('Saved oof_predictions.csv')

if __name__ == '__main__':
    main()
