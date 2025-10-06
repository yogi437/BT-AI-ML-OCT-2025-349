import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

train = pd.read_csv("engineered_train.csv")
test = pd.read_csv("engineered_test.csv")

X = train.drop("Survived", axis=1)
y = train["Survived"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
test_scaled = scaler.transform(test)

model = LogisticRegression(max_iter=200)
model.fit(X_scaled, y)
predictions = model.predict(test_scaled)

submission = pd.DataFrame({
    "PassengerId": pd.read_csv("test.csv")["PassengerId"],
    "Survived": predictions
})

submission.to_csv("submission_logreg.csv", index=False)
print("✅ Submission file created: submission_logreg.csv")
