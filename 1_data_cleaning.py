# 1_data_cleaning.py
"""Load train/test, basic EDA (plots), and clean common missing values.
Save merged CSV as data_merged.csv for downstream steps.
Usage: python 1_data_cleaning.py
Requires: pandas, matplotlib
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

TRAIN = "train.csv"
TEST = "test.csv"
OUT_MERGED = "data_merged.csv"
EDA_DIR = "eda_plots"

def quick_eda(df):
    os.makedirs(EDA_DIR, exist_ok=True)
    # 1) Distribution of SalePrice (only train rows)
    if 'SalePrice' in df.columns:
        plt.figure()
        df['SalePrice'].dropna().hist(bins=50)
        plt.title("SalePrice distribution")
        plt.xlabel("SalePrice")
        plt.ylabel("Count")
        plt.savefig(os.path.join(EDA_DIR, "saleprice_distribution.png"))
        plt.close()
    # 2) Log SalePrice distribution
    if 'SalePrice' in df.columns:
        plt.figure()
        np.log1p(df['SalePrice'].dropna()).hist(bins=50)
        plt.title("Log1p(SalePrice) distribution")
        plt.xlabel("log1p(SalePrice)")
        plt.ylabel("Count")
        plt.savefig(os.path.join(EDA_DIR, "saleprice_log_distribution.png"))
        plt.close()
    # 3) Scatter: GrLivArea vs SalePrice (train only if available)
    if set(['GrLivArea','SalePrice']).issubset(df.columns):
        plt.figure()
        plt.scatter(df.loc[df['SalePrice']>0,'GrLivArea'], df.loc[df['SalePrice']>0,'SalePrice'], s=8)
        plt.title("GrLivArea vs SalePrice")
        plt.xlabel("GrLivArea")
        plt.ylabel("SalePrice")
        plt.savefig(os.path.join(EDA_DIR, "grlivarea_vs_saleprice.png"))
        plt.close()
    # 4) Missing values barplot (top 30)
    miss = df.isnull().sum()
    miss = miss[miss>0].sort_values(ascending=False)
    if len(miss)>0:
        top = miss.head(30)
        plt.figure(figsize=(8,6))
        top.plot(kind='bar')
        plt.title("Top missing value counts (top 30)")
        plt.ylabel("Missing count")
        plt.tight_layout()
        plt.savefig(os.path.join(EDA_DIR, "missing_values_top30.png"))
        plt.close()

def fill_missing(df):
    # Fill NA that mean 'None'
    none_cols = [
        "PoolQC","MiscFeature","Alley","Fence","FireplaceQu",
        "GarageType","GarageFinish","GarageQual","GarageCond",
        "BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2",
        "MasVnrType"
    ]
    for c in none_cols:
        if c in df.columns:
            df[c] = df[c].fillna("None")
    # Numeric columns where NA means 0
    zero_cols = ["GarageYrBlt","GarageArea","GarageCars",
                 "BsmtFinSF1","BsmtFinSF2","BsmtUnfSF","TotalBsmtSF",
                 "BsmtFullBath","BsmtHalfBath","MasVnrArea"]
    for c in zero_cols:
        if c in df.columns:
            df[c] = df[c].fillna(0)
    # LotFrontage: fill by median of neighborhood then global median
    if "LotFrontage" in df.columns and "Neighborhood" in df.columns:
        df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage']                                  .transform(lambda x: x.fillna(x.median()))
        df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())
    # Remaining numeric: median
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for c in num_cols:
        if df[c].isnull().sum() > 0:
            df[c] = df[c].fillna(df[c].median())
    # Remaining categorical: mode
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    for c in cat_cols:
        if df[c].isnull().sum() > 0:
            df[c] = df[c].fillna(df[c].mode()[0])
    return df

def main():
    print("This script expects train.csv and test.csv in the same folder.")
    train = pd.read_csv(TRAIN)
    test = pd.read_csv(TEST)
    print("Train shape:", train.shape)
    print("Test shape:", test.shape)
    # Quick EDA on train
    quick_eda(train)
    # Merge for preprocessing pipeline
    train['is_train'] = 1
    test['is_train'] = 0
    test['SalePrice'] = 0  # placeholder so columns align
    df = pd.concat([train, test], sort=False).reset_index(drop=True)
    print("Merged shape:", df.shape)
    df = fill_missing(df)
    df.to_csv(OUT_MERGED, index=False)
    print("Saved merged cleaned data to", OUT_MERGED)
    print("EDA plots saved in", EDA_DIR)

if __name__ == "__main__":
    main()
