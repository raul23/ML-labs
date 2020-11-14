"""Kaggle - Intro to Machine Learning (Exercises)

The Iowa House Prices dataset can be downloaded from
https://www.kaggle.com/nickptaylor/iowa-house-prices
"""
import os

import ipdb
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from ml_labs.utils.genutils import print_

# Path of the file to read
iowa_file_path = os.path.expanduser('~/Data/kaggle_datasets/iowa_house_prices/train.csv')


# Exercises 2 and 3: Basic Data Exploration and Your First Machine Learning Model
def ex_2_and_3():
    home_data = pd.read_csv(iowa_file_path)

    # Print the list of columns in the dataset to find the name of the prediction target
    print_("Columns", 0)
    print_(home_data.columns)

    # Select the target variable, which corresponds to the sales price.
    y = home_data.SalePrice
    print_("y", 0)
    print_(y)

    # Create the list of features below
    feature_names = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

    # Select data corresponding to features in feature_names
    X = home_data[feature_names]
    print_("X", 0)
    print_(X)

    # Review data
    # Print description or statistics from X
    print_("Summary of X", 0)
    print_(X.describe())

    # Print the top few lines
    print_("First few rows of X", 0)
    print_(X.head())

    # Specify the model: DecisionTreeRegressor
    # For model reproducibility, set a numeric value for random_state when specifying the model
    iowa_model = DecisionTreeRegressor(random_state=1)

    # Fit the model using the data in X and y
    iowa_model.fit(X, y)

    # Make predictions using X as the data
    print_("Making predictions on all houses from X:", 0)
    print_(X)
    print_("The predictions are", 0)
    print_(iowa_model.predict(X))

    # Print the top few lines of y
    print_("First few rows of y", 0)
    print_(y.head())


# Exercise 4: Model Validation
def ex_4():
    # --------------------------------------------------------------------
    # Set up your coding environment where the previous exercise left off.
    # --------------------------------------------------------------------
    home_data = pd.read_csv(iowa_file_path)
    y = home_data.SalePrice
    feature_columns = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
    X = home_data[feature_columns]

    # Specify Model
    iowa_model = DecisionTreeRegressor()
    # Fit Model
    iowa_model.fit(X, y)

    print("First in-sample predictions:", iowa_model.predict(X.head()))
    print("Actual target values for those homes:", y.head().tolist(), end="\n\n")

    # -----------------
    # Start of exercise
    # -----------------
    # Step 1: Split Your Data
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

    # Step 2: Specify and Fit the Model
    # Create a DecisionTreeRegressor model and fit it to the relevant data.

    # Specify the model
    iowa_model = DecisionTreeRegressor(random_state=1)

    # Fit iowa_model with the training data (from the splitting).
    iowa_model.fit(train_X, train_y)

    # Step 3: Make Predictions with Validation data
    # Predict with all validation observations
    val_predictions = iowa_model.predict(val_X)

    # Print the top few validation predictions
    print_("Top few predictions from validation data", 0)
    print_(val_predictions[:5].tolist())
    # Print the top few actual prices from validation data
    print_("Top few actual prices from validation data", 0)
    print_(val_y.head().to_list())

    # Step 4: Calculate the Mean Absolute Error in Validation Data
    val_mae = mean_absolute_error(val_y, val_predictions)
    print_("Mean Absolute Error in validation data", 0)
    print_(val_mae)


if __name__ == '__main__':
    # ex_2_and_3()
    ex_4()
