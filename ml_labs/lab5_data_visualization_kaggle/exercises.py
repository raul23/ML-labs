"""Kaggle - Data Visualization (Exercises)

The datasets can all be downloaded from the course's notebook.
"""
import os

import ipdb
import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
import seaborn as sns

from ml_labs.utils.genutils import print_

fifa_filepath = os.path.expanduser('~/Data/kaggle_datasets/fifa_kaggle_course/fifa.csv')
museum_filepath = os.path.expanduser('~/Data/kaggle_datasets/museum_visitors_kaggle_course/museum_visitors.csv')


# Exercise 1: Hello, Seaborn
def ex_1():
    print_("Exercise 1: Hello, Seaborn", 0, 1)

    # ---------------------
    # Step 2: Load the data
    # ---------------------
    fifa_data = pd.read_csv(fifa_filepath, index_col="Date", parse_dates=True)

    # ---------------------
    # Step 3: Plot the data
    # ---------------------
    # Set the width and height of the figure
    plt.figure(figsize=(16, 6))

    # Line chart showing how FIFA rankings evolved over time
    sns.lineplot(data=fifa_data)

    plt.show()


# Exercise 2: Line Charts
def ex_2():
    print_("Exercise 2: Line Charts", 0, 1)

    # ---------------------
    # Step 1: Load the data
    # ---------------------
    museum_data = pd.read_csv(museum_filepath, index_col="Date", parse_dates=True)

    # -----------------------
    # Step 2: Review the data
    # -----------------------
    # Print the last five rows of the data
    print_("Last 5 rows", 0)
    print_(museum_data.tail())

    # How many visitors did the Chinese American Museum receive in July 2018?
    ca_museum_jul18 = museum_data.loc['2018-07-01', 'Chinese American Museum']
    print_("Number of visitor the Chinese American Museum receive in July 2018", 0)
    print_(ca_museum_jul18)

    # In October 2018, how many more visitors did Avila
    # Adobe receive than the Firehouse Museum?
    subset = museum_data.loc['2018-10-01', ['Avila Adobe', 'Firehouse Museum']]
    avila_oct18 = subset[0] - subset[1]
    print_("Number of visitors Avila Adobe received more than the Firehouse Museum (October 2018)", 0)
    print_(avila_oct18)

    # ---------------------------------
    # Step 3: Convince the museum board
    # ---------------------------------
    # Set the width and height of the figure
    plt.figure(figsize=(14, 6))

    # Add title
    plt.title("Monthly visitors for 4 museums in LA")

    # Line chart showing number of visitors to each museum over time
    sns.lineplot(data=museum_data)

    plt.show()

    # --------------------------
    # Step 4: Assess seasonality
    # --------------------------
    # Part A
    # Line plot showing the number of visitors to Avila Adobe over time
    # Set the width and height of the figure
    plt.figure(figsize=(14, 6))

    # Add title
    plt.title("Monthly visitors to Avila Adobe museum")

    # Line chart showing number of visitors to Avila Adobe over time
    sns.lineplot(data=museum_data['Avila Adobe'], label="Avila Adobe")

    # Add label for horizontal axis
    plt.xlabel("Date")

    plt.show()


if __name__ == '__main__':
    # ex_1()
    ex_2()
    # ex_3()
    # ex_4()
    # ex_5()
    # ex_6()
    # ex_7()
    # ex_8()
