import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv("cleaned_train.csv")

sns.countplot(x='Survived', data=train)
plt.title('Survival Count')
plt.show()

sns.countplot(x='Sex', hue='Survived', data=train)
plt.title('Survival by Gender')
plt.show()

sns.countplot(x='Pclass', hue='Survived', data=train)
plt.title('Survival by Passenger Class')
plt.show()

train['Age'].hist(bins=20)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()
