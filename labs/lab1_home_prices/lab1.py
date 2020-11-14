# Ref.: Kaggle - Intro to ML
import os
import pandas as pd
import ipdb


def print_(msg):
    print("*** {} ***".format(msg))


# Load Melbourne Housing Snapshot dataset
melbourne_file_path = os.path.expanduser('~/Data/kaggle_datasets/melbourne_housing_snapshot/melb_data.csv')
melbourne_data = pd.read_csv(melbourne_file_path)
# Print a summary of the data in Melbourne data
print_("Summary of dataset")
print(melbourne_data.describe(), end="\n\n")

# List of all columns in the dataset
print_("Columns")
print(melbourne_data.columns, end="\n\n")

# Drop missing values
melbourne_data = melbourne_data.dropna(axis=0)

# Select the prediction target (price)
y = melbourne_data.Price
print_("y")
print(y, end="\n\n")

# Select features
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = melbourne_data[melbourne_features]
print_("X")
print(X, end="\n\n")

print_("Summary of X")
print(X.describe(), end="\n\n")
print_("First few rows of X")
print(X.head(), end="\n\n")
