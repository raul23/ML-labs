"""Kaggle - Intro to Machine Learning (Tutorial)

The Melbourne Housing Snapshot dataset can be downloaded from
https://www.kaggle.com/glovepm/melbourne-housing
"""
import os

import ipdb
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from ml_labs.utils.genutils import print_


# Load Melbourne Housing Snapshot dataset
melbourne_file_path = os.path.expanduser('~/Data/kaggle_datasets/melbourne_housing_snapshot/melb_data.csv')
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
