"""Kaggle - Pandas (Tutorials)

The Wine Reviews dataset can be download from
https://www.kaggle.com/zynicide/wine-reviews
"""
import os

import ipdb
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


# Exercise 2: Indexing, Selecting & Assigning
def lesson_2():
    print_("Lesson 2: Indexing, Selecting & Assigning", 0, 1)
    reviews = pd.read_csv(wine_file_path, index_col=0)
    pd.set_option('max_rows', 5)

    # ----------------
    # Native accessors
    # ----------------
    print_("Country column from reviews", 0)
    print_(reviews.country)  # also reviews['country']

    print_("First country from the country Series", 0)
    print_(reviews.country[0])

    # ------------------
    # Indexing in pandas
    # ------------------
    # pandas' own accessor operators: loc and iloc
    #
    # NOTE: loc and iloc are row-first, column-second
    # This is the opposite of what we do in native Python, which is
    # column-first, row-second.

    # Index-based selection: iloc
    # NOTE 1: iloc requires numeric indexers,
    # NOTE 2: iloc indexes exclusively
    # Select the first row of data in a DataFrame
    print_("First row of data", 0)
    print_(reviews.iloc[0])

    print_("Get the first column from a DataFrame", 0)
    print_(reviews.iloc[:, 0])

    print_("Get the first 3 rows from the country column", 0)
    print_(reviews.iloc[:3, 0])

    print_("Get the 2nd and 3rd rows from the country column", 0)
    print_(reviews.iloc[1:3, 0])

    print_("Get the first 3 rows from the country column using a list", 0)
    print_(reviews.iloc[[0, 1, 2], 0])

    print_("Get the 5 last elements from the dataset", 0)
    print_(reviews.iloc[-5:])

    # Label-based selection: loc
    # NOTE 1: loc works with string indexers,
    # NOTE 2: loc, meanwhile, indexes inclusively
    print_("Get the first entry in reviews (using loc)", 0)
    print_(reviews.loc[0, 'country'])

    print_("Get columns from the dataset using loc", 0)
    print_(reviews.loc[:, ['taster_name', 'taster_twitter_handle', 'points']])

    # ----------------------
    # Manipulating the index
    # ----------------------
    print_("set_index to the title field", 0)
    print_(reviews.set_index("title"))

    # ---------------------
    # Conditional selection
    # ---------------------
    print_("Check if each wine is Italian or not", 0)
    print_(reviews.country == 'Italy')

    print_("Get italian wined", 0)
    print_(reviews.loc[reviews.country == 'Italy'])

    # AND: &
    print_("Get italian wines that are better than average", 0)
    print_(reviews.loc[(reviews.country == 'Italy') & (reviews.points >= 90)])

    # OR : | (pipe)
    print_("Get italian or better than average wines", 0)
    print_(reviews.loc[(reviews.country == 'Italy') | (reviews.points >= 90)])

    # isin conditional selector
    print_("Get wines from Italy or France", 0)
    print_(reviews.loc[reviews.country.isin(['Italy', 'France'])])

    # isnull and notnull: is (or not) empty (NaN)
    print_("Get wines with a price tag", 0)
    print_(reviews.loc[reviews.price.notnull()])

    # --------------
    # Assigning data
    # --------------
    # Assign a constant value
    # Every row gets 'everyone'
    reviews['critic'] = 'everyone'
    print_("Assign a constant value", 0)
    print_(reviews['critic'])

    # Assign an iterable of values
    reviews['index_backwards'] = range(len(reviews), 0, -1)
    print_("Assign an iterable of values", 0)
    print_(reviews['index_backwards'])


if __name__ == '__main__':
    # lesson_1()
    lesson_2()
