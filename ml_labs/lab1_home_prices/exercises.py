"""Kaggle - Intro to Machine Learning (Exercises)

The Iowa House Prices dataset can be downloaded from
https://www.kaggle.com/nickptaylor/iowa-house-prices
"""
import os

import ipdb
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from ml_labs.utils.genutils import print_


# Path of the file to read
iowa_file_path = os.path.expanduser('~/Data/kaggle_datasets/iowa_house_prices/train.csv')

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
