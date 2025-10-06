import pandas as pd

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

train["Embarked"].fillna(train["Embarked"].mode()[0], inplace=True)
test["Fare"].fillna(test["Fare"].median(), inplace=True)

train.to_csv("cleaned_train.csv", index=False)
test.to_csv("cleaned_test.csv", index=False)

print("✅ Missing values handled successfully!")
