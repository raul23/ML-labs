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


# Exercise 4: Grouping and Sorting
def ex_4():
    print_("Exercise 4: Grouping and Sorting", 0, 1)
    reviews = pd.read_csv(wine_130k_file_path, index_col=0)

    # ---------
    # Problem 1
    # ---------
    # Who are the most common wine reviewers in the dataset? Create a Series
    # whose index is the taster_twitter_handle category from the dataset, and
    # whose values count how many reviews each person wrote.
    print_("Problem 1", 0)
    reviews_written = reviews.groupby(['taster_twitter_handle']).description.agg([len]).iloc[:, 0]
    print_(reviews_written)

    # Other answers:
    # - reviews_written = reviews.groupby('taster_twitter_handle').size()
    # - reviews_written = reviews.groupby('taster_twitter_handle').taster_twitter_handle.count()

    # ---------
    # Problem 2
    # ---------
    # What is the best wine I can buy for a given amount of money? Create a
    # Series whose index is wine prices and whose values is the maximum number
    # of points a wine costing that much was given in a review. Sort the values
    # by price, ascending (so that 4.0 dollars is at the top and 3300.0 dollars
    # is at the bottom).
    # best_rating_per_price = reviews.groupby('price').points.agg([max]).iloc[:, 0]
    print_("Problem 2", 0)
    best_rating_per_price = reviews.groupby('price').points.max()
    print_(best_rating_per_price)

    # ---------
    # Problem 3
    # ---------
    # What are the minimum and maximum prices for each variety of wine? Create a
    # DataFrame whose index is the variety category from the dataset and whose
    # values are the min and max values thereof.
    print_("Problem 3", 0)
    price_extremes = reviews.groupby('variety').price.agg([min, max])
    print_(price_extremes)

    # ---------
    # Problem 4
    # ---------
    # What are the most expensive wine varieties? Create a variable
    # sorted_varieties containing a copy of the dataframe from the previous
    # question where varieties are sorted in descending order based on minimum
    # price, then on maximum price (to break ties).
    print_("Problem 4", 0)
    sorted_varieties = price_extremes.sort_values(by=['min', 'max'], ascending=False)
    print_(sorted_varieties)

    # ---------
    # Problem 5
    # ---------
    # Create a Series whose index is reviewers and whose values is the average
    # review score given out by that reviewer. Hint: you will need the
    # taster_name and points columns.
    print_("Problem 5", 0)
    reviewer_mean_ratings = reviews.groupby('taster_name').points.mean()
    print_(reviewer_mean_ratings)

    # ---------
    # Problem 6
    # ---------
    # What combination of countries and varieties are most common? Create a
    # Series whose index is a MultiIndexof {country, variety} pairs. For
    # example, a pinot noir produced in the US should map to
    # {"US", "Pinot Noir"}. Sort the values in the Series in descending order
    # based on wine count.
    # My answer was not accepted but ...
    # country_variety_counts = reviews.groupby(['country', 'variety']).variety.agg([len]).sort_values('len', ascending=False).iloc[:, 0]
    print_("Problem 6", 0)
    country_variety_counts = reviews.groupby(['country', 'variety']).variety.size().sort_values(0, ascending=False)
    print_(country_variety_counts)


# Exercise 5: Data Types and Missing Values
def ex_5():
    print_("Exercise 5: Data Types and Missing Values", 0, 1)
    reviews = pd.read_csv(wine_130k_file_path, index_col=0)

    # ---------
    # Problem 1
    # ---------
    # What is the data type of the points column in the dataset?
    print_("Problem 1", 0)
    dtype = reviews.points.dtype
    print_(dtype)

    # ---------
    # Problem 2
    # ---------
    # Create a Series from entries in the points column, but convert the
    # entries to strings. Hint: strings are str in native Python.
    print_("Problem 2", 0)
    point_strings = reviews.points.astype('str')
    print_(point_strings)

    # ---------
    # Problem 3
    # ---------
    # Sometimes the price column is null. How many reviews in the dataset are
    # missing a price?
    print_("Problem 3", 0)
    n_missing_prices = reviews.price.isnull().sum()
    print_(n_missing_prices)

    # Other answers:
    # -  len(reviews[reviews.price.isnull()])
    # - pd.isnull(reviews.price).sum()

    # ---------
    # Problem 4
    # ---------
    # What are the most common wine-producing regions? Create a Series counting
    # the number of times each value occurs in the region_1 field. This field
    # is often missing data, so replace missing values with Unknown. Sort in
    # descending order. Your output should look something like this:
    """
    Unknown                    21247
    Napa Valley                 4480
                               ...  
    Bardolino Superiore            1
    Primitivo del Tarantino        1
    Name: region_1, Length: 1230, dtype: int64
    """
    print_("Problem 4", 0)
    # Ref.: https://stackoverflow.com/a/50270796 (for grouping a Series without lambda)
    reviews_per_region = reviews.region_1.fillna('Unknown').to_frame(0).groupby(0)[0].size().sort_values(ascending=False)
    print_(reviews_per_region)
    # Better solution:
    # reviews.region_1.fillna('Unknown').value_counts().sort_values(ascending=False)
    # Thus, value_counts() <=> to_frame(0).groupby(0)[0].size()


# Exercise 6: Renaming and Combining
def ex_6():
    print_("Exercise 6: Renaming and Combining", 0, 1)

    # ---------
    # Problem 1
    # ---------
    print_("Problem 1", 0)

    # ---------
    # Problem 2
    # ---------
    print_("Problem 2", 0)

    # ---------
    # Problem 3
    # ---------
    print_("Problem 3", 0)

    # ---------
    # Problem 4
    # ---------
    print_("Problem 4", 0)


if __name__ == '__main__':
    # ex_1()
    # ex_2()
    # ex_3()
    # ex_4()
    ex_5()
    # ex_6()
