"""Kaggle - Pandas (Exercises)
"""
import os

import ipdb
import pandas as pd

from ml_labs.utils.genutils import print_

wine_first150k_file_path = os.path.expanduser('~/Data/kaggle_datasets/wine_reviews/winemag-data_first150k.csv')
wine_130k_file_path = os.path.expanduser('~/Data/kaggle_datasets/wine_reviews/winemag-data-130k-v2.csv')


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
    reviews = pd.read_csv(wine_first150k_file_path, index_col=0)
    print_("Problem 4", 0)
    print_(reviews.head())

    # ---------
    # Problem 5
    # ---------
    animals = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
    animals.to_csv('cows_and_goats.csv')
    print_("Problem 4")
    print_(animals.head())


# Exercise 2: Indexing, Selecting & Assigning
def ex_2():
    pd.set_option("display.max_rows", 5)
    print_("Exercise 2: Indexing, Selecting & Assigning", 0, 1)
    reviews = pd.read_csv(wine_130k_file_path, index_col=0)

    print_("First 5 rows from reviews", 0)
    print_(reviews.head())

    # ---------
    # Problem 1
    # ---------
    # Select the description column from reviews and assign the result to the
    # variable desc
    desc = reviews.description
    print_("Problem 1", 0)
    print_(desc)

    # ---------
    # Problem 2
    # ---------
    # Select the first value from the description column of reviews, assigning
    # it to variable first_description
    first_description = desc[0]
    # Other answers: reviews.description.iloc[0] and reviews.description.loc[0]
    print_("Problem 2", 0)
    print_(first_description)

    # ---------
    # Problem 3
    # ---------
    # Select the first row of data (the first record) from reviews, assigning
    # it to the variable first_row
    first_row = reviews.iloc[0]
    # Other answer: reviews.loc[0]
    print_("Problem 3", 0)
    print_(first_row)

    # ---------
    # Problem 4
    # ---------
    # Select the first 10 values from the description column in reviews,
    # assigning the result to variable first_descriptions
    first_descriptions = reviews.loc[:9, 'description']
    # Other answers: reviews.description.iloc[:10], desc.head(10)
    print_("Problem 4", 0)
    print_(first_descriptions)

    # ---------
    # Problem 5
    # ---------
    # Select the records with index labels 1, 2, 3, 5, and 8, assigning the
    # result to the variable sample_reviews
    sample_reviews = reviews.iloc[[1, 2, 3, 5, 8]]
    print_("Problem 5", 0)
    print_(sample_reviews)

    # ---------
    # Problem 6
    # ---------
    # Create a variable df containing the country, province, region_1, and
    # region_2 columns of the records with the index labels 0, 1, 10, and 100
    df = reviews.loc[[0, 1, 10, 100], ['country', 'province', 'region_1', 'region_2']]
    print_("Problem 6", 0)
    print_(df)

    # ---------
    # Problem 7
    # ---------
    # Create a variable df containing the country and variety columns of the
    # first 100 records
    df = reviews.loc[:99, ['country', 'variety']]
    # Other answer: reviews.iloc[:100, [0, 11]]
    print_("Problem 7", 0)
    print_(df)

    # ---------
    # Problem 8
    # ---------
    # Create a DataFrame italian_wines containing reviews of wines made in Italy
    italian_wines = reviews.loc[reviews.country == 'Italy']
    print_("Problem 8", 0)
    print_(italian_wines)

    # ---------
    # Problem 9
    # ---------
    # Create a DataFrame top_oceania_wines containing all reviews with at least
    # 95 points (out of 100) for wines from Australia or New Zealand
    countries_cond = reviews.country.isin(['Australia', 'New Zealand'])
    points_cond = reviews.points >= 95
    top_oceania_wines = reviews.loc[countries_cond & points_cond]
    print_("Problem 9", 0)
    print_(top_oceania_wines)


# Exercise 3: Summary Functions and Maps
def ex_3():
    pd.set_option("display.max_rows", 5)
    print_("Exercise 3: Summary Functions and Maps", 0, 1)

    reviews = pd.read_csv(wine_130k_file_path, index_col=0)
    print_("First 5 rows from reviews", 0)
    print_(reviews.head())

    # ---------
    # Problem 1
    # ---------
    # What is the median of the points column in the reviews DataFrame?
    median_points = reviews.points.median()
    print_("Problem 1", 0)
    print_(median_points)

    # ---------
    # Problem 2
    # ---------
    # What countries are represented in the dataset? (Your answer should not
    # include any duplicates.)
    countries = reviews.country.unique()
    print_("Problem 2", 0)
    print_(countries)

    # ---------
    # Problem 3
    # ---------
    # How often does each country appear in the dataset? Create a Series
    # reviews_per_country mapping countries to the count of reviews of wines
    # from that country.
    reviews_per_country = reviews.country.value_counts()
    print_("Problem 3", 0)
    print_(reviews_per_country)

    # ---------
    # Problem 4
    # ---------
    # Create variable centered_price containing a version of the price
    # column with the mean price subtracted.
    centered_price = reviews.price - reviews.price.mean()
    print_("Problem 4", 0)
    print_(centered_price)

    # ---------
    # Problem 5
    # ---------
    # I'm an economical wine buyer. Which wine is the "best bargain"? Create a
    # variable bargain_wine with the title of the wine with the highest
    # points-to-price ratio in the dataset.
    ratio = reviews.points / reviews.price
    index = ratio.argmax()
    bargain_wine = reviews.loc[index].title
    print_("Problem 5", 0)
    print_(bargain_wine)

    # ---------
    # Problem 6
    # ---------
    # There are only so many words you can use when describing a bottle of wine.
    # Is a wine more likely to be "tropical" or "fruity"? Create a Series
    # descriptor_counts counting how many times each of these two words appears
    # in the description column in the dataset.
    word_counts = {'tropical': 0, 'fruity': 0}
    for desc in reviews.description:
        for word, count in word_counts.items():
            # If you want to compute the number of occurrences of each word
            # word_counts[word] += desc.count(word)
            if desc.count(word):
                word_counts[word] += 1

    descriptor_counts = pd.Series(word_counts)
    print_("Problem 6", 0)
    print_(descriptor_counts)

    # Other answer:
    """
    n_trop = reviews.description.map(lambda desc: "tropical" in desc).sum()
    n_fruity = reviews.description.map(lambda desc: "fruity" in desc).sum()
    descriptor_counts = pd.Series([n_trop, n_fruity], index=['tropical', 'fruity'])
    """

    # ---------
    # Problem 7
    # ---------
    # A score of 95 or higher counts as 3 stars, a score of at least 85 but less
    # than 95 is 2 stars. Any other score is 1 star.
    #
    # Also, the Canadian Vintners Association bought a lot of ads on the site,
    # so any wines from Canada should automatically get 3 stars, regardless of
    # points.
    #
    # Create a series star_ratings with the number of stars corresponding to
    # each review in the dataset.
    def convert_points(row):
        if row.country == 'Canada':
            row.points = 3
        elif row.points >= 95:
            row.points = 3
        elif row.points >= 85:
            row.points = 2
        else:
            row.points = 1
        return row

    star_ratings = reviews.apply(convert_points, axis='columns').points
    print_("Problem 7", 0)
    print_(star_ratings)


if __name__ == '__main__':
    # ex_1()
    # ex_2()
    ex_3()
