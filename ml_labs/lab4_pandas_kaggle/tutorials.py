"""Kaggle - Pandas (Tutorials)

The Wine Reviews dataset can be download from
https://www.kaggle.com/zynicide/wine-reviews
"""
import os

import pandas as pd

from ml_labs.utils.genutils import print_

wine_file_path = os.path.expanduser('~/Data/kaggle_datasets/wine_reviews/winemag-data-130k-v2.csv')


# Lesson 1: Creating, Reading and Writing
def lesson_1():
    print_("Lesson 1: Creating, Reading and Writing", 0, 1)
    # -------------
    # Creating data
    # -------------
    # DataFrame
    dt_int = pd.DataFrame({'Yes': [50, 21], 'No': [131, 2]})
    print_("Simple DataFrame with integers", 0)
    print_(dt_int)

    dt_str = pd.DataFrame({'Bob': ['I liked it.', 'It was awful.'],
                           'Sue': ['Pretty good.', 'Bland.']})
    print_("Simple DataFrame with strings", 0)
    print_(dt_str)

    dt_index = pd.DataFrame({'Bob': ['I liked it.', 'It was awful.'],
                             'Sue': ['Pretty good.', 'Bland.']},
                            index=['Product A', 'Product B'])
    print_("DataFrame with row labels", 0)
    print_(dt_index)

    # Series
    s_list = pd.Series([1, 2, 3, 4, 5])
    print_("Simple series with integers", 0)
    print_(s_list)

    # NOTE: a Series does not have a column name, it only has one overall name
    s_index_name = pd.Series([30, 35, 40],
                             index=['2015 Sales', '2016 Sales', '2017 Sales'],
                             name='Product A')
    print_("Series with row labels and a name", 0)
    print_(s_index_name)

    # ---------------
    # Read data files
    # ---------------
    wine_reviews = pd.read_csv(wine_file_path)
    print_("How large the Wine Reviews dataset is", 0)
    print("Shape: ", wine_reviews.shape)
    print("Number of entries: ", wine_reviews.shape[0] * wine_reviews.shape[1])
    print()

    print_("First 5 rows from the Wine Reviews dataset", 0)
    print_(wine_reviews.head())

    # Make pandas use the CSV's built-in index for the index (instead of
    # creating a new one from scratch) by specifying an index_col
    wine_reviews = pd.read_csv(wine_file_path, index_col=0)
    print_("First 5 rows from the Wine Reviews dataset [using index_col=0]")
    print_(wine_reviews.head())


if __name__ == '__main__':
    lesson_1()
