"""Kaggle - Intro to Machine Learning (Bonus Lessons)

The Titanic dataset can be downloaded from
https://www.kaggle.com/c/titanic
"""
import os

import ipdb
import pandas as pd

from ml_labs.utils.genutils import print_

titanic_file_path = os.path.expanduser('~/Data/kaggle_datasets/titanic/train.csv')


# Bonus Lesson 2: Getting Started With Titanic
def titanic():
    train_data = pd.read_csv(titanic_file_path)
    print_("First 5 rows from Titanic dataset", 0)
    print_(train_data.head())


if __name__ == '__main__':
    titanic()
