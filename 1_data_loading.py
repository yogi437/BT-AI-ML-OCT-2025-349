import pandas as pd

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

print("Train shape:", train.shape)
print("Test shape:", test.shape)
print(train.head())

print("\nMissing values in training data:")
print(train.isnull().sum())

train.to_csv("train_backup.csv", index=False)
test.to_csv("test_backup.csv", index=False)
