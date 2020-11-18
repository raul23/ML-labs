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
ign_filepath = os.path.expanduser('~/Data/kaggle_datasets/ign_scores_kaggle_course/ign_scores.csv')


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


# Exercise 3: Bar Charts and Heatmaps
def ex_3():
    print_("Exercise 3: Bar Charts and Heatmaps", 0, 1)
    # ---------------------
    # Step 1: Load the data
    # ---------------------
    ign_data = pd.read_csv(ign_filepath, index_col="Platform")

    # -----------------------
    # Step 2: Review the data
    # -----------------------
    # Print the data
    print_("The whole data", 0)
    print_(ign_data)

    # What is the highest average score received by PC games, for any platform?
    high_score = 7.759930

    # On the Playstation Vita platform, which genre has the lowest average score?
    # Please provide the name of the column, and put your answer in single quotes
    # (e.g., 'Action', 'Adventure', 'Fighting', etc.)
    worst_genre = 'Simulation'

    # -------------------------------
    # Step 3: Which platform is best?
    # -------------------------------
    # Part A
    # Create a bar chart that shows the average score for racing games, for each
    # platform. Your chart should have one bar for each platform.

    # Bar chart showing average score for racing games by platform
    # Set the width and height of the figure
    plt.figure(figsize=(10, 6))

    # Add title
    plt.title("Average score for racing games by platform")

    # Bar chart showing average score for racing games by platform
    sns.barplot(x=ign_data.index, y=ign_data['Racing'])

    # Add label for vertical axis
    plt.ylabel("Average score")

    plt.show()

    # ----------------------------------
    # Step 4: All possible combinations!
    # ----------------------------------
    # Part A
    # Heatmap showing average game score by platform and genre
    # Set the width and height of the figure
    plt.figure(figsize=(14, 7))

    # Add title
    plt.title("Average game score by platform and genre")

    # Heatmap showingaverage game score by platform and genre
    sns.heatmap(data=ign_data, annot=True)

    # Add label for horizontal axis
    plt.xlabel("Platform")

    plt.show()


if __name__ == '__main__':
    # ex_1()
    # ex_2()
    ex_3()
    # ex_4()
    # ex_5()
    # ex_6()
    # ex_7()
    # ex_8()
