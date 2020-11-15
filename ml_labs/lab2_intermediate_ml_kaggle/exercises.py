"""Kaggle - Intermediate Machine Learning (Exercises)

The Home Prices dataset can be downloaded from
https://www.kaggle.com/c/home-data-for-ml-course/data
"""
import os

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from ml_labs.utils.genutils import print_

train_file_path = os.path.expanduser('~/Data/kaggle_datasets/home_data_for_ml_course/train.csv')
test_file_path = os.path.expanduser('~/Data/kaggle_datasets/home_data_for_ml_course/train.csv')


# Exercise 1: Introduction
def ex_1():
    # Read the data
    X_full = pd.read_csv(train_file_path, index_col='Id')
    X_test_full = pd.read_csv(test_file_path, index_col='Id')

    # Obtain target and predictors
    y = X_full.SalePrice
    features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
    X = X_full[features].copy()
    X_test = X_test_full[features].copy()

    # Break off validation set from training data
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                          random_state=0)

    print_("First 5 rows from the train dataset", 0)
    print_(X_train.head())

    # -------------------------------
    # Step 1: Evaluate several models
    # -------------------------------
    # Define five different random forest models
    model_1 = RandomForestRegressor(n_estimators=50, random_state=0)
    model_2 = RandomForestRegressor(n_estimators=100, random_state=0)
    model_3 = RandomForestRegressor(n_estimators=100, criterion='mae', random_state=0)
    model_4 = RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=0)
    model_5 = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=0)

    models = [model_1, model_2, model_3, model_4, model_5]

    # Function for comparing different models
    def score_model(model, X_t=X_train, X_v=X_valid, y_t=y_train, y_v=y_valid):
        model.fit(X_t, y_t)
        preds = model.predict(X_v)
        return mean_absolute_error(y_v, preds)

    for i in range(0, len(models)):
        mae = score_model(models[i])
        print("Model %d MAE: %d" % (i + 1, mae))

    # Fill in the best model
    best_model = model_3

    # ---------------------------------
    # Step 2: Generate test predictions
    # ---------------------------------
    # Create a Random Forest model
    my_model = RandomForestRegressor(n_estimators=100, criterion='mae', random_state=0)
    # Fit the model to the training data
    my_model.fit(X, y)

    # Generate test predictions
    preds_test = my_model.predict(X_test)

    # Save predictions in format used for competition scoring
    output = pd.DataFrame({'Id': X_test.index,
                           'SalePrice': preds_test})
    output.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    ex_1()
