"""Kaggle - Data Visualization (Tutorials)

Datasets used for the tutorials:
- The FIFA dataset
- The Spotify dataset

They can all be downloaded from the course's notebook.
"""
import os

import ipdb
import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
import seaborn as sns

from ml_labs.utils.genutils import print_

fifa_filepath = os.path.expanduser('~/Data/kaggle_datasets/fifa_kaggle_course/fifa.csv')
spotify_filepath = os.path.expanduser('~/Data/kaggle_datasets/spotify_kaggle/spotify.csv')


# Lesson 1: Hello, Seaborn
def lesson_1():
    print_("Lesson 1: Hello, Seaborn", 0, 1)
    # -------------
    # Load the data
    # -------------
    fifa_data = pd.read_csv(fifa_filepath, index_col="Date", parse_dates=True)

    # ----------------
    # Examine the data
    # ----------------
    print_("The first 5 rows of the data", 0)
    print_(fifa_data.head())

    # -------------
    # Plot the data
    # -------------
    # Set the width and height of the figure
    plt.figure(figsize=(16, 6))

    # Line chart showing how FIFA rankings evolved over time
    sns.lineplot(data=fifa_data)

    plt.show()


# Lesson 2: Line Charts
def lesson_2():
    print_("Lesson 2: Line Charts", 0, 1)

    # -------------
    # Load the data
    # -------------
    spotify_data = pd.read_csv(spotify_filepath, index_col="Date", parse_dates=True)

    # ----------------
    # Examine the data
    # ----------------
    # Print the first 5 rows of the data
    print_("First 5 rows", 0)
    print_(spotify_data.head())

    # Print the last five rows of the data
    print_("Last 5 rows", 0)
    print_(spotify_data.tail())

    # -------------
    # Plot the data
    # -------------
    # Line chart showing daily global streams of each song
    sns.lineplot(data=spotify_data)
    plt.show()


if __name__ == '__main__':
    # lesson_1()
    lesson_2()
    # lesson_3()
    # lesson_4()
    # lesson_5()
    # lesson_6()
    # lesson_7()
    # lesson_8()
