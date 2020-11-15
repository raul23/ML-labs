"""Kaggle - Intermediate Machine Learning (Tutorials)

The Melbourne Housing Snapshot dataset can be downloaded from
https://www.kaggle.com/glovepm/melbourne-housing
"""
import os

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from ml_labs.utils.genutils import print_

melbourne_file_path = os.path.expanduser('~/Data/kaggle_datasets/melbourne_housing_snapshot/melb_data.csv')


# Measure Quality of Each Approach
def score_model(model, X_t, X_v, y_t, y_v):
    model.fit(X_t, y_t)
    preds = model.predict(X_v)
    return mean_absolute_error(y_v, preds)


# Lesson 2: Missing values
def lesson_2():
    # ----------------------------------
    # Example: Melbourne Housing dataset
    # ----------------------------------
    # Load data
    melbourne_data = pd.read_csv(melbourne_file_path)
    # Choose target and features
    y = melbourne_data.Price
    melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea',
                          'YearBuilt', 'Lattitude', 'Longtitude']
    X = melbourne_data[melbourne_features]
    # Split data into training and validation data,
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=0)

    # Build a random forest model
    forest_model = RandomForestRegressor(random_state=1)

    # --------------------------------------------
    # Approach 1: Drop Columns with Missing Values
    # --------------------------------------------
    # Get names of columns with missing values
    cols_with_missing = [col for col in X_train.columns
                         if X_train[col].isnull().any()]

    # Drop columns in training and validation data
    reduced_X_train = X_train.drop(cols_with_missing, axis=1)
    reduced_X_valid = X_valid.drop(cols_with_missing, axis=1)

    print("MAE from Approach 1 (Drop columns with missing values):")
    print(score_model(forest_model, reduced_X_train, reduced_X_valid, y_train, y_valid))

    # ----------------------
    # Approach 2: Imputation
    # ----------------------
    # Imputation
    my_imputer = SimpleImputer()
    imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
    imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))

    # Imputation removed column names; put them back
    imputed_X_train.columns = X_train.columns
    imputed_X_valid.columns = X_valid.columns

    print("\nMAE from Approach 2 (Imputation):")
    print(score_model(forest_model, imputed_X_train, imputed_X_valid, y_train, y_valid))

    # --------------------------------------
    # Approach 3: An Extension to Imputation
    # --------------------------------------
    # We impute the missing values, while also keeping track of which values
    # were imputed

    # Make copy to avoid changing original data (when imputing)
    X_train_plus = X_train.copy()
    X_valid_plus = X_valid.copy()

    # Make new columns indicating what will be imputed
    for col in cols_with_missing:
        X_train_plus[col + '_was_missing'] = X_train_plus[col].isnull()
        X_valid_plus[col + '_was_missing'] = X_valid_plus[col].isnull()

    # Imputation
    my_imputer = SimpleImputer()
    imputed_X_train_plus = pd.DataFrame(my_imputer.fit_transform(X_train_plus))
    imputed_X_valid_plus = pd.DataFrame(my_imputer.transform(X_valid_plus))

    # Imputation removed column names; put them back
    imputed_X_train_plus.columns = X_train_plus.columns
    imputed_X_valid_plus.columns = X_valid_plus.columns

    print("\nMAE from Approach 3 (An Extension to Imputation):")
    print_(score_model(forest_model, imputed_X_train_plus, imputed_X_valid_plus, y_train, y_valid))

    # Shape of training data (num_rows, num_columns)
    print_("Shape of training data (num_rows, num_columns)", 0)
    print_(X_train.shape)

    # Number of missing values in each column of training data
    missing_val_count_by_column = (X_train.isnull().sum())
    print_("Number of missing values in each column of training data", 0)
    print_(missing_val_count_by_column[missing_val_count_by_column > 0])


if __name__ == '__main__':
    lesson_2()
