"""Kaggle - Pandas (Exercises)
"""
import os

import pandas as pd

from ml_labs.utils.genutils import print_

wine_file_path = os.path.expanduser('~/Data/kaggle_datasets/wine_reviews/winemag-data_first150k.csv')


# Exercise 1: Creating, Reading and Writing
def ex_1():
    print_("Exercise 1: Creating, Reading and Writing", 0, 1)
    # ---------
    # Problem 1
    # ---------
    fruits = pd.DataFrame({'Apples': [30], 'Bananas': [21]})
    print_("Problem 1", 0)
    print_(fruits)

    # ---------
    # Problem 2
    # ---------
    fruit_sales = pd.DataFrame({'Apples': [35, 41], 'Bananas': [21, 34]},
                               index=['2017 Sales', '2018 Sales'])
    print_("Problem 2", 0)
    print_(fruit_sales)

    # ---------
    # Problem 3
    # ---------
    ingredients = pd.Series(['4 cups', '1 cup', '2 large', '1 can'],
                            index=['Flour', 'Milk', 'Eggs', 'Spam'],
                            name='Dinner')
    print_("Problem 3", 0)
    print_(ingredients)

    # ---------
    # Problem 4
    # ---------
    reviews = pd.read_csv(wine_file_path, index_col=0)
    print_("Problem 4", 0)
    print_(reviews.head())

    # ---------
    # Problem 5
    # ---------
    animals = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
    animals.to_csv('cows_and_goats.csv')
    print_("Problem 4")
    print_(animals.head())


if __name__ == '__main__':
    ex_1()
