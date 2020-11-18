"""Kaggle - Data Visualization (Exercises)

Datasets used for the tutorials:
- The FIFA dataset
-

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


if __name__ == '__main__':
    ex_1()
    # ex_2()
    # ex_3()
    # ex_4()
    # ex_5()
    # ex_6()
    # ex_7()
    # ex_8()
