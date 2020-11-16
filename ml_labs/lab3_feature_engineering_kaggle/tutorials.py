"""Kaggle - Feature Engineering (Tutorials)

The Kickstarter Projects dataset can be downloaded from
https://www.kaggle.com/kemical/kickstarter-projects
"""
import os

import ipdb
import category_encoders as ce
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

from ml_labs.utils.genutils import print_

ks_projects_file_path = os.path.expanduser('~/Data/kaggle_datasets/kickstart_projects/ks-projects-201801.csv')


# Lesson 1: Baseline Model
def lesson_1():
    print_("Lesson 1: Baseline Model", 0, 1)
    ks = pd.read_csv(ks_projects_file_path,
                     parse_dates=['deadline', 'launched'])
    print_("First 6 rows from the Kickstarter Projects dataset", 0)
    print_(ks.head(6))

    print('Unique values in `state` column:', list(ks.state.unique()))

    # Prepare the target column
    # Drop live projects
    ks = ks.query('state != "live"')

    # Add outcome column, "successful" == 1, others are 0
    ks = ks.assign(outcome=(ks['state'] == 'successful').astype(int))

    # Convert timestamps
    ks = ks.assign(hour=ks.launched.dt.hour,
                   day=ks.launched.dt.day,
                   month=ks.launched.dt.month,
                   year=ks.launched.dt.year)

    # Prep categorical variables
    cat_features = ['category', 'currency', 'country']
    encoder = LabelEncoder()

    # Apply the label encoder to each column
    encoded = ks[cat_features].apply(encoder.fit_transform)

    # Collect all of these features in a new dataframe that we can use to train
    # a model
    #
    # Since ks and encoded have the same index and I can easily join them
    data = ks[['goal', 'hour', 'day', 'month', 'year', 'outcome']].join(encoded)
    data.head()

    # Create training, validation, and test splits
    # Use 10% of the data as a validation set, 10% for testing, and the other
    # 80% for training.
    valid_fraction = 0.1
    valid_size = int(len(data) * valid_fraction)

    train = data[:-2 * valid_size]
    valid = data[-2 * valid_size:-valid_size]
    test = data[-valid_size:]

    # Train a model
    feature_cols = train.columns.drop('outcome')

    dtrain = lgb.Dataset(train[feature_cols], label=train['outcome'])
    dvalid = lgb.Dataset(valid[feature_cols], label=valid['outcome'])

    param = {'num_leaves': 64, 'objective': 'binary'}
    param['metric'] = 'auc'
    num_round = 1000
    bst = lgb.train(param, dtrain, num_round, valid_sets=[dvalid], early_stopping_rounds=10, verbose_eval=False)

    # Make predictions & evaluate the model
    ypred = bst.predict(test[feature_cols])
    score = metrics.roc_auc_score(test['outcome'], ypred)

    print(f"Test AUC score: {score}")


# Lesson 2: Categorical Encodings
def lesson_2():
    print_("Lesson 2: Categorical Encodings", 0, 1)
    ks = pd.read_csv(ks_projects_file_path,
                     parse_dates=['deadline', 'launched'])

    # Drop live projects
    ks = ks.query('state != "live"')

    # Add outcome column, "successful" == 1, others are 0
    ks = ks.assign(outcome=(ks['state'] == 'successful').astype(int))

    # Timestamp features
    ks = ks.assign(hour=ks.launched.dt.hour,
                   day=ks.launched.dt.day,
                   month=ks.launched.dt.month,
                   year=ks.launched.dt.year)

    # Label encoding
    cat_features = ['category', 'currency', 'country']
    encoder = LabelEncoder()
    encoded = ks[cat_features].apply(encoder.fit_transform)

    data_cols = ['goal', 'hour', 'day', 'month', 'year', 'outcome']
    data = ks[data_cols].join(encoded)

    # Defining  functions that will help us test our encodings
    def get_data_splits(dataframe, valid_fraction=0.1):
        valid_fraction = 0.1
        valid_size = int(len(dataframe) * valid_fraction)

        train = dataframe[:-valid_size * 2]
        # valid size == test size, last two sections of the data
        valid = dataframe[-valid_size * 2:-valid_size]
        test = dataframe[-valid_size:]

        return train, valid, test

    def train_model(train, valid):
        feature_cols = train.columns.drop('outcome')

        dtrain = lgb.Dataset(train[feature_cols], label=train['outcome'])
        dvalid = lgb.Dataset(valid[feature_cols], label=valid['outcome'])

        param = {'num_leaves': 64, 'objective': 'binary',
                 'metric': 'auc', 'seed': 7, 'verbose': -1}
        bst = lgb.train(param, dtrain, num_boost_round=1000, valid_sets=[dvalid],
                        early_stopping_rounds=10, verbose_eval=False)

        valid_pred = bst.predict(valid[feature_cols])
        valid_score = metrics.roc_auc_score(valid['outcome'], valid_pred)
        print(f"Validation AUC score: {valid_score:.4f}")

    # Train a model (on the baseline data)
    train, valid, test = get_data_splits(data)
    print_("Baseline (LightGBM with no categorical encoding)", 0)
    train_model(train, valid)
    print()

    # --------------
    # Count Encoding
    # --------------
    cat_features = ['category', 'currency', 'country']

    # Create the encoder
    count_enc = ce.CountEncoder()

    # Transform the features, rename the columns with the _count suffix, and join to dataframe
    # TODO: calculating the counts on the whole dataset? Should it be on the train only to avoid data leakage?
    # This is what was done in the Exercise 2
    count_encoded = count_enc.fit_transform(ks[cat_features])
    data = data.join(count_encoded.add_suffix("_count"))

    # Train a model
    train, valid, test = get_data_splits(data)
    print_("LightGBM with COUNT encoding", 0)
    train_model(train, valid)
    print()

    # ---------------
    # Target Encoding
    # ---------------
    # Create the encoder
    target_enc = ce.TargetEncoder(cols=cat_features)
    target_enc.fit(train[cat_features], train['outcome'])

    # Transform the features, rename the columns with _target suffix, and join to dataframe
    train_TE = train.join(target_enc.transform(train[cat_features]).add_suffix('_target'))
    valid_TE = valid.join(target_enc.transform(valid[cat_features]).add_suffix('_target'))

    # Train a model
    print_("LightGBM with TARGET encoding", 0)
    train_model(train_TE, valid_TE)
    print()

    # -----------------
    # CatBoost Encoding
    # -----------------
    # Create the encoder
    cb_enc = ce.TargetEncoder(cols=cat_features)
    cb_enc.fit(train[cat_features], train['outcome'])

    # Transform the features, rename the columns with _target suffix, and join to dataframe
    train_CBE = train.join(cb_enc.transform(train[cat_features]).add_suffix('_cb'))
    valid_CBE = valid.join(cb_enc.transform(valid[cat_features]).add_suffix('_cb'))

    # Train a model
    print_("LightGBM with CatBoost encoding", 0)
    train_model(train_CBE, valid_CBE)
    print()


# Lesson 3: Feature Generation
def lesson_3():
    print_("Lesson 3: Feature Generation", 0, 1)
    # -----
    # Setup
    # -----
    ks = pd.read_csv(ks_projects_file_path,
                     parse_dates=['deadline', 'launched'])

    # Drop live projects
    ks = ks.query('state != "live"')

    # Add outcome column, "successful" == 1, others are 0
    ks = ks.assign(outcome=(ks['state'] == 'successful').astype(int))

    # Timestamp features
    ks = ks.assign(hour=ks.launched.dt.hour,
                   day=ks.launched.dt.day,
                   month=ks.launched.dt.month,
                   year=ks.launched.dt.year)

    # Label encoding
    cat_features = ['category', 'currency', 'country']
    encoder = LabelEncoder()
    encoded = ks[cat_features].apply(encoder.fit_transform)

    data_cols = ['goal', 'hour', 'day', 'month', 'year', 'outcome']
    baseline_data = ks[data_cols].join(encoded)

    # ------------
    # Interactions
    # ------------
    interactions = ks['category'] + "_" + ks['country']
    print_("Interactions: first 5 rows from category_country", 0)
    print_(interactions.head(5))

    # Label encode the interaction feature and add it to the data
    label_enc = LabelEncoder()
    data_interaction = baseline_data.assign(category_country=label_enc.fit_transform(interactions))
    print_("First 5 rows from data with the added interactions", 0)
    print_(data_interaction.head())

    # -----------------------------------
    # Number of projects in the last week
    # -----------------------------------
    # First, create a Series with a timestamp index
    launched = pd.Series(ks.index, index=ks.launched, name="count_7_days").sort_index()
    print_("First 20 rows from series with the timestamp index", 0)
    print_(launched.head(20))

    count_7_days = launched.rolling('7d').count() - 1
    print_("First 20 rows from the rolling window of 7 days", 0)
    print_(count_7_days.head(20))

    # Ignore records with broken launch dates
    plt.plot(count_7_days[7:])
    plt.title("Number of projects launched over periods of 7 days")
    plt.show()

    # Adjust the index so we can join it with the other training data.
    count_7_days.index = launched.values
    count_7_days = count_7_days.reindex(ks.index)

    print_("First 10 rows from the rolling window of 7 days (with index adjusted)", 0)
    print_(count_7_days.head(10))

    # Now join the new feature with the other data again using .join since
    # we've matched the index.
    print_("First 10 rows from baseline data with the new feature (count_7_days))")
    print_(baseline_data.join(count_7_days).head(10))

    # ------------------------------------------------
    # Time since the last project in the same category
    # ------------------------------------------------
    def time_since_last_project(series):
        # Return the time in hours
        return series.diff().dt.total_seconds() / 3600.

    df = ks[['category', 'launched']].sort_values('launched')
    timedeltas = df.groupby('category').transform(time_since_last_project)
    print_("First 20 rows from timedeltas (time since the last project in "
           "the same category)", 0)
    print_(timedeltas.head(20))
    # We get NaNs here for projects that are the first in their category.

    # Fix NaNs by using the mean or median. We'll also need to reset the index
    # so we can join it with the other data.
    # Final time since last project
    timedeltas = timedeltas.fillna(timedeltas.median()).reindex(baseline_data.index)
    print_("First 20 rows from timedeltas (with NaNs fixed)", 0)
    print_(timedeltas.head(20))

    # -------------------------------
    # Transforming numerical features
    # -------------------------------
    # Some models work better when the features are normally distributed
    # Transform them with the square root or natural logarithm.

    # Example: transform the goal feature using the square root and log functions

    # Square root transformation
    plt.hist(np.sqrt(ks.goal), range=(0, 400), bins=50)
    plt.title('Sqrt(Goal)')
    plt.show()

    # Log function transformation
    plt.hist(np.log(ks.goal), range=(0, 25), bins=50)
    plt.title('Log(Goal)')
    plt.show()

    # IMPORTANT: The log transformation won't help our model since tree-based
    # models are scale invariant. However, this should help if we had a linear
    # model or neural network.


if __name__ == '__main__':
    # lesson_1()
    # lesson_2()
    lesson_3()
