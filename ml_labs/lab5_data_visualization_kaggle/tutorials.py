"""Kaggle - Data Visualization (Tutorials)

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
flight_filepath = os.path.expanduser('~/Data/kaggle_datasets/flight_delays_kaggle_course/flight_delays.csv')
insurance_filepath = os.path.expanduser('~/Data/kaggle_datasets/insurance_kaggle_course/insurance.csv')
iris_filepath = os.path.expanduser('~/Data/kaggle_datasets/iris_kaggle_course/iris.csv')
iris_set_filepath = "~/Data/kaggle_datasets/iris_kaggle_course/iris_setosa.csv"
iris_ver_filepath = "~/Data/kaggle_datasets/iris_kaggle_course/iris_versicolor.csv"
iris_vir_filepath = "~/Data/kaggle_datasets/iris_kaggle_course/iris_virginica.csv"
spotify_filepath = os.path.expanduser('~/Data/kaggle_datasets/spotify_kaggle_course/spotify.csv')


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
    # Set the width and height of the figure
    plt.figure(figsize=(14, 6))

    # Add title
    plt.title("Daily Global Streams of Popular Songs in 2017-2018")

    # Line chart showing daily global streams of each song
    sns.lineplot(data=spotify_data)

    plt.show()

    # -------------------------
    # Plot a subset of the data
    # -------------------------
    # List of columns
    print_("Columns from the dataset", 0)
    print_(list(spotify_data.columns))

    # Set the width and height of the figure
    plt.figure(figsize=(14, 6))

    # Add title
    plt.title("Daily Global Streams of Popular Songs in 2017-2018")

    # Line chart showing daily global streams of 'Shape of You'
    sns.lineplot(data=spotify_data['Shape of You'], label="Shape of You")

    # Line chart showing daily global streams of 'Despacito'
    sns.lineplot(data=spotify_data['Despacito'], label="Despacito")

    # Add label for horizontal axis
    plt.xlabel("Date")

    plt.show()


# Lesson 3: Bar Charts and Heatmaps
def lesson_3():
    print_("Lesson 3: Bar Charts and Heatmaps", 0, 1)

    # -------------
    # Load the data
    # -------------
    flight_data = pd.read_csv(flight_filepath, index_col="Month")

    # ----------------
    # Examine the data
    # ----------------
    print_("The whole data", 0)
    print_(flight_data)

    # ---------
    # Bar chart
    # ---------
    # Set the width and height of the figure
    plt.figure(figsize=(10, 6))

    # Add title
    plt.title("Average Arrival Delay for Spirit Airlines Flights, by Month")

    # Bar chart showing average arrival delay for Spirit Airlines flights by month
    sns.barplot(x=flight_data.index, y=flight_data['NK'])

    # Add label for vertical axis
    plt.ylabel("Arrival delay (in minutes)")

    plt.show()

    # Important: You must select the indexing column with flight_data.index,
    # and it is not possible to use flight_data['Month'] (which will return an
    # error). This is because when we loaded the dataset, the "Month" column
    # was used to index the rows. We always have to use this special notation
    # to select the indexing column.

    # -------
    # Heatmap
    # -------
    # Set the width and height of the figure
    plt.figure(figsize=(14, 7))

    # Add title
    plt.title("Average Arrival Delay for Each Airline, by Month")

    # Heatmap showing average arrival delay for each airline by month
    # NOTE: annot=True - This ensures that the values for each cell appear on
    # the chart. (Leaving this out removes the numbers from each of the cells!)
    sns.heatmap(data=flight_data, annot=True)

    # Add label for horizontal axis
    plt.xlabel("Airline")

    plt.show()


# Lesson 4: Scatter Plots
def lesson_4():
    print_("Lesson 4: Scatter Plots", 0, 1)

    # -------------------------
    # Load and examine the data
    # -------------------------
    insurance_data = pd.read_csv(insurance_filepath)
    print_("First 5 rows", 0)
    print_(insurance_data.head())

    # -------------
    # Scatter plots
    # -------------
    sns.scatterplot(x=insurance_data['bmi'], y=insurance_data['charges'])
    plt.show()

    # Draw a line that best fits the data
    sns.regplot(x=insurance_data['bmi'], y=insurance_data['charges'])
    plt.show()

    # -------------------------
    # Color-coded scatter plots
    # -------------------------
    # Use color-coded scatter plots to display the relationships between 3
    # variables
    sns.scatterplot(x=insurance_data['bmi'], y=insurance_data['charges'], hue=insurance_data['smoker'])
    plt.show()

    # sns.lmplot: adds two regression lines
    sns.lmplot(x="bmi", y="charges", hue="smoker", data=insurance_data)
    plt.show()

    # ------------------------
    # Categorical scatter plot
    # ------------------------
    # We use this sort of scatter plot to highlight the relationship between a
    # continuous and categorical variables
    sns.swarmplot(x=insurance_data['smoker'],
                  y=insurance_data['charges'])
    plt.show()


# Lesson 5: Distributions
def lesson_5():
    print_("Lesson 5: Distributions", 0, 1)
    iris_data = pd.read_csv(iris_filepath, index_col="Id")

    # Print the first 5 rows of the data
    print_("First 5 rows", 0)
    print_(iris_data.head())

    # ----------
    # Histograms
    # ----------
    # Create a histogram to see how petal length varies in iris flowers
    # NOTE: kde=False is something we'll always provide when creating a
    # histogram, as leaving it out will create a slightly different plot.
    sns.distplot(a=iris_data['Petal Length (cm)'], kde=False)
    plt.show()

    # -------------
    # Density plots
    # -------------
    # kernel density estimate (KDE) plot: a kind of smoothed histogram
    # KDE plot
    # NOTE: shade=True colors the area below the curve
    sns.kdeplot(data=iris_data['Petal Length (cm)'], shade=True)
    plt.show()

    # ------------
    # 2D KDE plots
    # ------------
    # 2D KDE plot
    # darker parts of the figure are more likely
    sns.jointplot(x=iris_data['Petal Length (cm)'],
                  y=iris_data['Sepal Width (cm)'], kind="kde")
    plt.show()

    # -----------------
    # Color-coded plots
    # -----------------
    iris_set_data = pd.read_csv(iris_set_filepath, index_col="Id")
    iris_ver_data = pd.read_csv(iris_ver_filepath, index_col="Id")
    iris_vir_data = pd.read_csv(iris_vir_filepath, index_col="Id")

    # Print the first 5 rows of the Iris versicolor data
    print_("First 5 rows of the Iris versicolor data", 0)
    print_(iris_ver_data.head())

    # Histograms for each species
    sns.distplot(a=iris_set_data['Petal Length (cm)'], label="Iris-setosa", kde=False)
    sns.distplot(a=iris_ver_data['Petal Length (cm)'], label="Iris-versicolor", kde=False)
    sns.distplot(a=iris_vir_data['Petal Length (cm)'], label="Iris-virginica", kde=False)

    # Add title
    plt.title("Histogram of Petal Lengths, by Species")

    # Force legend to appear
    plt.legend()

    plt.show()

    # KDE plots for each species
    sns.kdeplot(data=iris_set_data['Petal Length (cm)'], label="Iris-setosa", shade=True)
    sns.kdeplot(data=iris_ver_data['Petal Length (cm)'], label="Iris-versicolor", shade=True)
    sns.kdeplot(data=iris_vir_data['Petal Length (cm)'], label="Iris-virginica", shade=True)

    # Add title
    plt.title("Distribution of Petal Lengths, by Species")

    # Force legend to appear
    plt.legend()

    plt.show()


if __name__ == '__main__':
    # lesson_1()
    # lesson_2()
    # lesson_3()
    # lesson_4()
    lesson_5()
    # lesson_6()
    # lesson_7()
    # lesson_8()
