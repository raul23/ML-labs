"""Kaggle - Intro to Machine Learning (Tutorial)

The Melbourne Housing Snapshot dataset can be downloaded from
https://www.kaggle.com/glovepm/melbourne-housing
"""
import os

import ipdb
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from ml_labs.utils.genutils import print_

melbourne_file_path = os.path.expanduser('~/Data/kaggle_datasets/melbourne_housing_snapshot/melb_data.csv')


# A utility function to help compare MAE scores from different values for max_leaf_nodes
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return mae


def lessons_1_to_3():
    # Load Melbourne Housing Snapshot dataset
    melbourne_data = pd.read_csv(melbourne_file_path)
    # Print a summary of the data in Melbourne data
    print_("Summary of dataset", 0)
    print_(melbourne_data.describe())

    # List of all columns in the dataset
    print_("Columns", 0)
    print_(melbourne_data.columns)

    # Drop missing values
    melbourne_data = melbourne_data.dropna(axis=0)

    # Select the prediction target (price)
    y = melbourne_data.Price
    print_("y", 0)
    print_(y)

    # Select features
    melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
    X = melbourne_data[melbourne_features]
    print_("X", 0)
    print_(X)

    print_("Summary of X", 0)
    print_(X.describe())
    print_("First few rows of X", 0)
    print_(X.head())

    # Define model. Specify a number for random_state to ensure same results each run
    melbourne_model = DecisionTreeRegressor(random_state=1)

    # Fit model
    melbourne_model.fit(X, y)

    print_("Making predictions for the following 5 houses:", 0)
    print_(X.head())
    print_("The predictions are", 0)
    print_(melbourne_model.predict(X.head()))


# Lesson 4: Model validation
def lesson_4():
    # Load data
    melbourne_data = pd.read_csv(melbourne_file_path)
    # Filter rows with missing price values
    filtered_melbourne_data = melbourne_data.dropna(axis=0)
    # Choose target and features
    y = filtered_melbourne_data.Price
    melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea',
                          'YearBuilt', 'Lattitude', 'Longtitude']
    X = filtered_melbourne_data[melbourne_features]

    # Define model
    melbourne_model = DecisionTreeRegressor()
    # Fit model
    melbourne_model.fit(X, y)

    # Calculate the mean absolute error
    predicted_home_prices = melbourne_model.predict(X)
    mae = mean_absolute_error(y, predicted_home_prices)
    print_("Mean absolute error when using just train set", 0)
    print_(mae)

    # Split data into training and validation data, for both features and target
    # The split is based on a random number generator. Supplying a numeric value to
    # the random_state argument guarantees we get the same split every time we
    # run this script.
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
    # Define model
    melbourne_model = DecisionTreeRegressor()
    # Fit model
    melbourne_model.fit(train_X, train_y)

    # Get predicted prices on validation data
    val_predictions = melbourne_model.predict(val_X)
    print_("Mean absolute error when using train and validation sets", 0)
    print_(mean_absolute_error(val_y, val_predictions))


# Lesson 5: Underfitting and Overfitting
def lesson_5():
    # -------------------------
    # Code from previous lesson
    # -------------------------
    # Load data
    melbourne_data = pd.read_csv(melbourne_file_path)
    # Filter rows with missing price values
    filtered_melbourne_data = melbourne_data.dropna(axis=0)
    # Choose target and features
    y = filtered_melbourne_data.Price
    melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea',
                          'YearBuilt', 'Lattitude', 'Longtitude']
    X = filtered_melbourne_data[melbourne_features]
    # Split data into training and validation data,
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

    # -----------------
    # Start of tutorial
    # -----------------
    # Compare MAE with differing values of max_leaf_nodes
    for max_leaf_nodes in [5, 50, 500, 5000]:
        my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
        print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" % (max_leaf_nodes, my_mae))


# Lesson 6: Random Forests
def lesson_6():
    # -------------------------
    # Code from previous lesson
    # -------------------------
    # Load data
    melbourne_data = pd.read_csv(melbourne_file_path)
    # Filter rows with missing price values
    filtered_melbourne_data = melbourne_data.dropna(axis=0)
    # Choose target and features
    y = filtered_melbourne_data.Price
    melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea',
                          'YearBuilt', 'Lattitude', 'Longtitude']
    X = filtered_melbourne_data[melbourne_features]
    # Split data into training and validation data,
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

    # -----------------
    # Start of tutorial
    # -----------------
    # Build a random forest model
    forest_model = RandomForestRegressor(random_state=1)
    forest_model.fit(train_X, train_y)
    melb_preds = forest_model.predict(val_X)
    print_("MAE for a Random Forest", 0)
    print_(mean_absolute_error(val_y, melb_preds))


if __name__ == '__main__':
    # lessons_1_to_3()
    # lesson_4()
    # lesson_5()
    lesson_6()
