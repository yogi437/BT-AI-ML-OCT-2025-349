import pandas as pd
from sklearn.preprocessing import LabelEncoder

train = pd.read_csv("cleaned_train.csv")
test = pd.read_csv("cleaned_test.csv")

combine = [train, test]

for dataset in combine:
    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    dataset['Title'] = dataset['Title'].replace(['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace(['Mlle','Ms'], 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    dataset['Sex'] = dataset['Sex'].map({'male': 0, 'female': 1})
    dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    dataset['Age'].fillna(dataset['Age'].median(), inplace=True)
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    dataset.drop(['Cabin', 'Ticket', 'Name'], axis=1, inplace=True)

le = LabelEncoder()
train['Title'] = le.fit_transform(train['Title'])
test['Title'] = le.transform(test['Title'])

train.to_csv("engineered_train.csv", index=False)
test.to_csv("engineered_test.csv", index=False)

print("✅ Feature engineering complete!")
