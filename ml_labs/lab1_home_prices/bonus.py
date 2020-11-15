"""Kaggle - Intro to Machine Learning (Bonus Lessons)

The Titanic dataset can be downloaded from
https://www.kaggle.com/c/titanic
"""
import os

import ipdb
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from ml_labs.utils.genutils import print_

titanic_train_file_path = os.path.expanduser('~/Data/kaggle_datasets/titanic/train.csv')
titanic_test_file_path = os.path.expanduser('~/Data/kaggle_datasets/titanic/test.csv')


# Bonus Lesson 2: Getting Started With Titanic
def titanic():
    # Load Titanic train dataset
    train_data = pd.read_csv(titanic_train_file_path)
    print_("First 5 rows from Titanic train dataset", 0)
    print_(train_data.head())

    # Load test set
    test_data = pd.read_csv(titanic_test_file_path)
    print_("First 5 rows from Titanic test dataset", 0)
    print_(test_data.head())

    # Part 3: Improve your score
    # Explore a pattern: assume that all female passengers survived (and all
    # male passengers died)
    women = train_data.loc[train_data.Sex == 'female']["Survived"]
    rate_women = sum(women) / len(women)

    # Based on the train set
    print("% of women who survived:", rate_women)

    men = train_data.loc[train_data.Sex == 'male']["Survived"]
    rate_men = sum(men) / len(men)

    # Based on the train set
    print("% of men who survived:", rate_men)

    # Your first machine learning model: a random forest model
    y = train_data["Survived"]

    features = ["Pclass", "Sex", "SibSp", "Parch"]
    X = pd.get_dummies(train_data[features])
    X_test = pd.get_dummies(test_data[features])

    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
    model.fit(X, y)
    predictions = model.predict(X_test)

    output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
    output.to_csv('my_submission.csv', index=False)
    print("Your submission was successfully saved!")


if __name__ == '__main__':
    titanic()
