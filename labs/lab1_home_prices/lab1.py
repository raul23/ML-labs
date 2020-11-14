# Ref.: Kaggle - Intro to ML
import os
import pandas as pd


# Load Melbourne Housing Snapshot dataset
melbourne_file_path = os.path.expanduser('~/Data/kaggle_datasets/melbourne_housing_snapshot/melb_data.csv')
melbourne_data = pd.read_csv(melbourne_file_path)
# Print a summary of the data in Melbourne data
print(melbourne_data.describe(), end="\n\n")

# List of all columns in the dataset
print(melbourne_data.columns, end="\n\n")

# Drop missing values
melbourne_data = melbourne_data.dropna(axis=0)

# Select the prediction target (price)
y = melbourne_data.Price
print(y, end="\n\n")

# Select features
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = melbourne_data[melbourne_features]
print(X, end="\n\n")
