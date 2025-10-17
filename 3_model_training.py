# 3_model_training.py
"""Train baseline LinearRegression, RandomForest, and XGBoost with K-Fold CV reporting.
Saves best model (by CV mean RMSE) as best_model.joblib and feature_importances.csv for tree models.
Usage: python 3_model_training.py
Requires: pandas, scikit-learn, xgboost, joblib, numpy
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import joblib
from math import sqrt

DATA_PRE = "data_preprocessed.csv"
OUT_MODEL = "best_model.joblib"
FI_CSV = "feature_importances.csv"

def rmse_cv(model, X, y, cv):
    scores = cross_val_score(model, X, y, scoring='neg_root_mean_squared_error', cv=cv, n_jobs=-1)
    return -scores.mean(), scores.std()

def load_data():
    df = pd.read_csv(DATA_PRE)
    train = df[df['is_train']==1].copy()
    test = df[df['is_train']==0].copy()
    if 'Id' in df.columns:
        test_ids = test['Id']
    else:
        test_ids = None
    X = train.drop(columns=['SalePrice','is_train','Id'], errors='ignore')
    y = train['SalePrice']
    X_test = test.drop(columns=['SalePrice','is_train','Id'], errors='ignore')
    return X, y, X_test, test_ids

def main():
    X, y, X_test, test_ids = load_data()
    print("Train shape:", X.shape, y.shape)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    models = {
        'LinearRegression': LinearRegression(),
        'RandomForest': RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        'XGBoost': XGBRegressor(n_estimators=500, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
    }
    results = {}
    for name, model in models.items():
        print(f"Evaluating {name}...")
        mean_rmse, std_rmse = rmse_cv(model, X, y, cv)
        print(f"{name} CV RMSE (log scale): {mean_rmse:.5f} ± {std_rmse:.5f}")
        results[name] = (mean_rmse, std_rmse)
    # Choose best model by mean RMSE
    best_name = min(results.items(), key=lambda x: x[1][0])[0]
    best_model = models[best_name]
    print("Fitting best model:", best_name)
    best_model.fit(X, y)
    joblib.dump(best_model, OUT_MODEL)
    print("Saved best model to", OUT_MODEL)
    # If tree-based, save feature importances
    if hasattr(best_model, 'feature_importances_'):
        fi = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False)
        fi.to_csv(FI_CSV, header=['importance'])
        print("Saved feature importances to", FI_CSV)

if __name__ == '__main__':
    main()
