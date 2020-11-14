# Ref.: Kaggle - Intro to ML
import os
import pandas as pd


# Load Melbourne Housing Snapshot dataset
melbourne_file_path = os.path.expanduser('~/Data/kaggle_datasets/melbourne_housing_snapshot/melb_data.csv')
melbourne_data = pd.read_csv(melbourne_file_path)
# Print a summary of the data in Melbourne data
print(melbourne_data.describe())
# List of all columns in the dataset
print(melbourne_data.columns)
# Drop missing values
melbourne_data = melbourne_data.dropna(axis=0)
