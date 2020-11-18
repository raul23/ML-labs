"""Kaggle - Pandas (Tutorials)

The Wine Reviews dataset can be download from
https://www.kaggle.com/zynicide/wine-reviews
"""
import os

import ipdb
import numpy as np
import pandas as pd
pd.set_option('max_rows', 5)

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


# Exercise 3: Summary Functions and Maps
def lesson_3():
    print_("Lesson 3: Summary Functions and Maps", 0, 1)
    reviews = pd.read_csv(wine_file_path, index_col=0)
    print_("Reviews", 0)
    print_(reviews)

    # -----------------
    # Summary functions
    # -----------------
    # Describe with numerical data
    print_("Describe reviews.points (numerical data only)", 0)
    print_(reviews.points.describe())

    # Describe with string data
    print_("Describe reviews.taster_name (string data)", 0)
    print_(reviews.taster_name.describe())

    # Statistic: mean
    print_("Mean of reviews.points", 0)
    print_(reviews.points.mean())

    # Unique values
    print_("Unique values from reviews.taster_name", 0)
    print_(reviews.taster_name.unique())

    # Unique values and how often they occur in the dataset
    print_("Unique values and their counts from reviews.taster_name", 0)
    print_(reviews.taster_name.value_counts())

    # ----
    # Maps
    # ----
    # Two important mapping methods: map() and apply()
    # NOTE: they don't modify the original data they're called on

    # map()
    # Remean the scores the wines received to 0
    review_points_mean = reviews.points.mean()
    remeans = reviews.points.map(lambda p: p - review_points_mean)
    print_("Remean the wine scores to 0 using map()", 0)
    print_(remeans)

    # apply()
    # NOTE: apply() is way slower than map()
    def remean_points(row):
        row.points = row.points - review_points_mean
        return row

    # NOTE: if axis='index', we transform each column
    # Commented because too slow
    """
    reviews_remeans = reviews.apply(remean_points, axis='columns')
    print_("Remean the wine scores to 0 using apply()", 0)
    print_(reviews_remeans.points)
    """

    # Faster way to remeaning the points column
    review_points_mean = reviews.points.mean()
    remeans = reviews.points - review_points_mean
    print_("Remean the wine scores to 0 using .mean() [Faster]", 0)
    print_(remeans)

    # Combining columns
    comb_cols = reviews.country + " - " + reviews.region_1
    print_("Combining country and region info", 0)
    print_(comb_cols)

    # IMPORTANT:
    # These operators (e.g. -, >) are faster than map() or apply()
    # because they uses speed ups built into pandas
    #
    # Though map() or apply() are more flexible becausE they can do more
    # advanced things, like applying conditional logic, which cannot be done
    # with addition and subtraction alone.


# Lesson 4: Grouping and Sorting
def lesson_4():
    pd.set_option("display.max_rows", 5)
    print_("Lesson 4: Grouping and Sorting", 0, 1)
    reviews = pd.read_csv(wine_file_path, index_col=0)

    # ------------------
    # Groupwise analysis
    # ------------------
    print_("Count occurrences of each point using group_by()", 0)
    print_(reviews.groupby('points').points.count())

    # Equivalent to using value_counts()
    print_("Count occurrences of each point using value_counts()", 0)
    print_(reviews.points.value_counts().sort_index())

    # Get the cheapest wine in each point value category
    print_("Cheapest wine in each point value category", 0)
    print_(reviews.groupby('points').price.min())

    # Select the name of the first wine reviewed from each winery
    print_("Select the name of the first wine reviewed from each winery using apply()", 0)
    print_(reviews.groupby('winery').apply(lambda df: df.title.iloc[0]))

    # You can also group by more than one column
    # Example: pick out the best wine by country and province:
    print_("Pick out the best wine by country and province", 0)
    print_(reviews.groupby(['country', 'province']).apply(lambda df: df.loc[df.points.idxmax()]))

    # agg(): lets you run a bunch of different functions on your DataFrame simultaneously
    # Example: generate a simple statistical summary of the dataset by country
    print_("Statistical summary by country", 0)
    print_(reviews.groupby(['country']).price.agg([len, min, max]))

    # -------------
    # Multi-indexes
    # -------------
    # vs single-level (regular) indices
    # More info about multi-indexes at https://pandas.pydata.org/pandas-docs/stable/advanced.html
    countries_reviewed = reviews.groupby(['country', 'province']).description.agg([len])
    print_("Multi-index: country and province", 0)
    print_(countries_reviewed)

    # reset_index(): important multi-index method that converts back to a
    # regular index
    print_("reset_index(): get back to the original single index", 0)
    print_(countries_reviewed.reset_index())

    # -------
    # Sorting
    # -------
    countries_reviewed = countries_reviewed.reset_index()
    print_("Sort by 'len' (ascending)", 0)
    print_(countries_reviewed.sort_values(by='len'))

    print_("Sort by 'len' (descending)", 0)
    print_(countries_reviewed.sort_values(by='len', ascending=False))

    # Sort by index values
    print_("Sort by index values", 0)
    print_(countries_reviewed.sort_index())

    # Sort by more than one column at a time
    print_("Sort by 2 columns: country and len", 0)
    countries_reviewed.sort_values(by=['country', 'len'])


# Lesson 5: Data Types and Missing Values
def lesson_5():
    # pd.set_option('max_rows', 5)
    print_("Lesson 5: Data Types and Missing Values", 0, 1)
    reviews = pd.read_csv(wine_file_path, index_col=0)

    # ------
    # DTypes
    # ------
    # column.dtype
    print_("dtype of the price column", 0)
    print_(reviews.price.dtype)

    # DataFrame.dtypes: dtypes of every column
    print_("dtypes of every column", 0)
    print_(reviews.dtypes)

    # object type: for strings

    # astype(): converts a column of one type into another
    print_("Convert points from int64 t float64", 0)
    print_(reviews.points.astype('float64'))

    # ------------
    # Missing data
    # ------------
    # NaN values are always of the float64 dtype

    # Select NaN entries
    print_("Select NaN entries for country", 0)
    print_(reviews[pd.isnull(reviews.country)])

    # Replace missing values with fillna()
    print_("Replace missing values with Unknown", 0)
    print_(reviews.region_2.fillna("Unknown"))

    # Backfill strategy for filling missing values: fill each missing value
    # with the first non-null value that appears sometime after the given
    # record in the database.

    # Replace a non-null value: replace()
    print_("Replace @kerinokeefe to @kerino", 0)
    print_(reviews.taster_twitter_handle.replace("@kerinokeefe", "@kerino"))


# Lesson 6: Renaming and Combining
def lesson_6():
    # pd.set_option('max_rows', 5)
    print_("Lesson 6: Renaming and Combining", 0, 1)
    reviews = pd.read_csv(wine_file_path, index_col=0)

    # --------
    # Renaming
    # --------
    # rename(): lets you change index names and/or column names

    # Change column
    # Change the points column in our dataset to score
    print_("Change the points column to score", 0)
    print_(reviews.rename(columns={'points': 'score'}))

    # Change indexes
    print_("Rename some elements of the index", 0)
    print_(reviews.rename(index={0: 'firstEntry', 1: 'secondEntry'}))

    # IMPORTANT: set_index() is usually more convenient than using rename()
    # to change indexes

    # rename_axis(): change the names for the row index and the column index
    print_("Change the row index to wines and the column index to fields", 0)
    print_(reviews.rename_axis("wines", axis='rows').rename_axis("fields", axis='columns'))

    # ---------
    # Combining
    # ---------
    # Three core methods for combining DataFrames and Series (start less complex)
    # - concat()
    # - join()
    # - merge()
    #
    # NOTE: what merge() can do, join() can do it more simply

    # concat(): smush a given list of elements together along an axis
    #
    # Smush two datasets
    # Ref.: https://www.kaggle.com/datasnaek/youtube-new
    canadian_youtube = pd.read_csv(os.path.expanduser("~/Data/kaggle_datasets/trending_youtube/CAvideos.csv"))
    british_youtube = pd.read_csv(os.path.expanduser("~/Data/kaggle_datasets/trending_youtube/GBvideos.csv"))

    print_("Concat two datasets", 0)
    print_(pd.concat([canadian_youtube, british_youtube]))

    # join(): lets you combine different DataFrame objects which have an index
    # in common
    #
    # Pull down videos that happened to be trending on the same day in both
    # Canada and the UK
    print_("videos that happened to be trending on the same day in both Canada "
           "and the UK", 0)
    left = canadian_youtube.set_index(['title', 'trending_date'])
    right = british_youtube.set_index(['title', 'trending_date'])

    print_(left.join(right, lsuffix='_CAN', rsuffix='_UK'))


if __name__ == '__main__':
    # lesson_1()
    # lesson_2()
    # lesson_3()
    # lesson_4()
    # lesson_5()
    lesson_6()
