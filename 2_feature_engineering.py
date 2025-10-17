# 2_feature_engineering.py
"""Create derived features, encode categoricals (one-hot for simplicity),
and save data_preprocessed.csv for modeling.
Usage: python 2_feature_engineering.py
Requires: pandas, numpy, scikit-learn, matplotlib
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler

IN_MERGED = "data_merged.csv"
OUT_PRE = "data_preprocessed.csv"
ENCODING_INFO = "encoding_info.txt"
PLOTS_DIR = "fe_plots"

def add_basic_features(df):
    # Total square footage (if components exist)
    if set(['TotalBsmtSF','1stFlrSF','2ndFlrSF']).issubset(df.columns):
        df['TotalSqFeet'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    # House age at sale
    if 'YearBuilt' in df.columns and 'YrSold' in df.columns:
        df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    # Total bathrooms approx
    if any(c in df.columns for c in ['FullBath','BsmtFullBath','HalfBath','BsmtHalfBath']):
        df['TotalBath'] = 0
        if 'FullBath' in df.columns: df['TotalBath'] += df['FullBath']
        if 'BsmtFullBath' in df.columns: df['TotalBath'] += df['BsmtFullBath']
        if 'HalfBath' in df.columns: df['TotalBath'] += df['HalfBath']*0.5
        if 'BsmtHalfBath' in df.columns: df['TotalBath'] += df['BsmtHalfBath']*0.5
    return df

def encode_and_scale(df):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    # Log-transform target for train rows
    if 'SalePrice' in df.columns:
        df.loc[df['is_train']==1, 'SalePrice'] = df.loc[df['is_train']==1, 'SalePrice'].apply(lambda x: np.log1p(x))
    # Save cardinality info for top categorical variables
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    with open(ENCODING_INFO, "w") as f:
        f.write("Categorical columns and top value counts:\n")
        for c in cat_cols:
            f.write(f"{c}: {df[c].nunique()} unique values\n")
    # Use one-hot encoding (drop_first to avoid dummy trap)
    df_encoded = pd.get_dummies(df, drop_first=True)
    # Scale numeric features (except target, is_train, Id)
    scaler = StandardScaler()
    numeric_cols = df_encoded.select_dtypes(include=[np.number]).columns.tolist()
    exclude = ['SalePrice','is_train','Id']
    numeric_cols = [c for c in numeric_cols if c not in exclude]
    df_encoded[numeric_cols] = scaler.fit_transform(df_encoded[numeric_cols])
    # Save a quick plot of feature counts
    plt.figure(figsize=(6,4))
    df_encoded.shape
    plt.bar([0], [df_encoded.shape[1]])
    plt.title("Number of features after encoding")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "feature_count.png"))
    plt.close()
    return df_encoded

def main():
    print("Loading merged data...")
    df = pd.read_csv(IN_MERGED)
    print("Initial shape:", df.shape)
    df = add_basic_features(df)
    df = encode_and_scale(df)
    df.to_csv(OUT_PRE, index=False)
    print("Saved preprocessed data to", OUT_PRE)
    print("Encoding info saved to", ENCODING_INFO)
    print("FE plots saved in", PLOTS_DIR)

if __name__ == "__main__":
    main()
