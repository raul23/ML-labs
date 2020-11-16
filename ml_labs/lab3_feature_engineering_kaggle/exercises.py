"""Kaggle - Feature Engineering (Exercises)

The TalkingData AdTracking dataset can be downloaded from
https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection
"""
import itertools
import os

import ipdb
import category_encoders as ce
import lightgbm as lgb
import pandas as pd
from sklearn import metrics
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

from ml_labs.utils.genutils import print_

adtracking_file_path = os.path.expanduser('~/Data/kaggle_datasets/talkingdata_adtracking_fraud_detection/train_sample.csv')


def load_data_for_ex_2():
    click_data = pd.read_csv(adtracking_file_path,
                             parse_dates=['click_time'])
    # print_("First 5 rows from the TalkingData AdTracking dataset", 0)
    # print_(click_data.head())

    # -------------------------------------
    # 1. Construct features from timestamps
    # -------------------------------------
    # Add new columns for timestamp features day, hour, minute, and second
    clicks = click_data.copy()
    clicks['day'] = clicks['click_time'].dt.day.astype('uint8')
    clicks['hour'] = clicks['click_time'].dt.hour.astype('uint8')
    clicks['minute'] = clicks['click_time'].dt.minute.astype('uint8')
    clicks['second'] = clicks['click_time'].dt.second.astype('uint8')
    return clicks


# Exercise 1: Baseline Model
def ex_1():
    print_("Exercise 1: Baseline Model", 0, 1)
    click_data = pd.read_csv(adtracking_file_path,
                             parse_dates=['click_time'])
    print_("First 5 rows from the TalkingData AdTracking dataset", 0)
    print_(click_data.head())

    # -------------------------------------
    # 1. Construct features from timestamps
    # -------------------------------------
    # Add new columns for timestamp features day, hour, minute, and second
    clicks = click_data.copy()
    clicks['day'] = clicks['click_time'].dt.day.astype('uint8')
    clicks['hour'] = clicks['click_time'].dt.hour.astype('uint8')
    clicks['minute'] = clicks['click_time'].dt.minute.astype('uint8')
    clicks['second'] = clicks['click_time'].dt.second.astype('uint8')

    # -----------------
    # 2. Label Encoding
    # -----------------
    cat_features = ['ip', 'app', 'device', 'os', 'channel']
    new_labels = [i + '_labels' for i in cat_features]

    # Create new columns in clicks using preprocessing.LabelEncoder()
    encoded = clicks[cat_features].apply(preprocessing.LabelEncoder().fit_transform)
    encoded.columns = new_labels
    clicks = clicks.join(encoded)
    print_("First 5 rows from the clicks data", 0)
    print_(clicks.head())

    # Create train/validation/test splits
    # IMPORTANT: this is time series data. Thus watch out for data leakage

    # The clicks DataFrame is sorted in order of increasing time. The first
    # 80% of the rows are the train set, the next 10% are the validation set,
    # and the last 10% are the test set.
    feature_cols = ['day', 'hour', 'minute', 'second',
                    'ip_labels', 'app_labels', 'device_labels',
                    'os_labels', 'channel_labels']

    valid_fraction = 0.1
    clicks_srt = clicks.sort_values('click_time')
    valid_rows = int(len(clicks_srt) * valid_fraction)
    train = clicks_srt[:-valid_rows * 2]
    # valid size == test size, last two sections of the data
    valid = clicks_srt[-valid_rows * 2:-valid_rows]
    test = clicks_srt[-valid_rows:]

    # Train with LightGBM
    dtrain = lgb.Dataset(train[feature_cols], label=train['is_attributed'])
    dvalid = lgb.Dataset(valid[feature_cols], label=valid['is_attributed'])
    dtest = lgb.Dataset(test[feature_cols], label=test['is_attributed'])

    param = {'num_leaves': 200, 'objective': 'binary'}
    param['metric'] = 'auc'
    num_round = 1000
    bst = lgb.train(param, dtrain, num_round, valid_sets=[dvalid], early_stopping_rounds=10)

    # Evaluate the model
    ypred = bst.predict(test[feature_cols])
    score = metrics.roc_auc_score(test['is_attributed'], ypred)
    print(f"Test score: {score}")


# Exercise 2: Categorical Encodings
def ex_2():
    print_("Exercise 2: Categorical Encodings", 0, 1)
    clicks = load_data_for_ex_2()

    def get_data_splits(dataframe, valid_fraction=0.1):
        """Splits a dataframe into train, validation, and test sets.

        First, orders by the column 'click_time'. Set the size of the
        validation and test sets with the valid_fraction keyword argument.
        """

        dataframe = dataframe.sort_values('click_time')
        valid_rows = int(len(dataframe) * valid_fraction)
        train = dataframe[:-valid_rows * 2]
        # valid size == test size, last two sections of the data
        valid = dataframe[-valid_rows * 2:-valid_rows]
        test = dataframe[-valid_rows:]

        return train, valid, test

    def train_model(train, valid, test=None, feature_cols=None):
        if feature_cols is None:
            feature_cols = train.columns.drop(['click_time', 'attributed_time',
                                               'is_attributed'])
        dtrain = lgb.Dataset(train[feature_cols], label=train['is_attributed'])
        dvalid = lgb.Dataset(valid[feature_cols], label=valid['is_attributed'])

        param = {'num_leaves': 256, 'objective': 'binary',
                 'metric': 'auc', 'seed': 7, 'verbose': -1}
        num_round = 1000
        bst = lgb.train(param, dtrain, num_round, valid_sets=[dvalid],
                        early_stopping_rounds=20, verbose_eval=False)

        valid_pred = bst.predict(valid[feature_cols])
        valid_score = metrics.roc_auc_score(valid['is_attributed'], valid_pred)
        print(f"Validation AUC score: {valid_score}")

        if test is not None:
            test_pred = bst.predict(test[feature_cols])
            test_score = metrics.roc_auc_score(test['is_attributed'], test_pred)
            return bst, valid_score, test_score
        else:
            return bst, valid_score

    print_("Baseline model", 0)
    train, valid, test = get_data_splits(clicks)
    _ = train_model(train, valid)
    print()

    # ------------------------------------
    # 1. Categorical encodings and leakage
    # ------------------------------------

    # ------------------
    # 2. Count encodings
    # ------------------
    cat_features = ['ip', 'app', 'device', 'os', 'channel']
    train, valid, test = get_data_splits(clicks)

    # Create the count encoder
    count_enc = ce.CountEncoder(cols=cat_features)

    # Learn encoding from the training set
    # TODO: Why not train['is_attributed']?
    count_enc.fit(train[cat_features])
    # count_enc.fit(train[cat_features], train['is_attributed'])

    # Apply encoding to the train and validation sets as new columns
    # Make sure to add `_count` as a suffix to the new columns
    train_encoded = train.join(count_enc.transform(train[cat_features]).add_suffix('_count'))
    valid_encoded = valid.join(count_enc.transform(valid[cat_features]).add_suffix('_count'))

    # Train the model on the encoded datasets
    print_("LightGBM with COUNT encoding", 0)
    _ = train_model(train_encoded, valid_encoded)
    print()

    # ------------------
    # 4. Target encoding
    # ------------------
    # Create the target encoder.
    target_enc = ce.TargetEncoder(cols=cat_features)

    # Learn encoding from the training set. Use the 'is_attributed' column as the target.
    target_enc.fit(train[cat_features], train['is_attributed'])

    # Apply encoding to the train and validation sets as new columns
    # Make sure to add `_target` as a suffix to the new columns
    train_encoded = train.join(target_enc.transform(train[cat_features]).add_suffix('_target'))
    valid_encoded = valid.join(target_enc.transform(valid[cat_features]).add_suffix('_target'))

    # Train a model
    print_("LightGBM with TARGET encoding", 0)
    _ = train_model(train_encoded, valid_encoded)
    print()

    # --------------------
    # 6. CatBoost Encoding
    # --------------------
    # Remove IP from the encoded features
    cat_features = ['app', 'device', 'os', 'channel']

    # Create the CatBoost encoder
    cb_enc = ce.CatBoostEncoder(cols=cat_features, random_state=7)

    # Learn encoding from the training set
    cb_enc.fit(train[cat_features], train['is_attributed'])

    # Apply encoding to the train and validation sets as new columns
    # Make sure to add `_cb` as a suffix to the new columns
    train_encoded = train.join(cb_enc.transform(train[cat_features]).add_suffix('_cb'))
    valid_encoded = valid.join(cb_enc.transform(valid[cat_features]).add_suffix('_cb'))

    # Train a model
    print_("LightGBM with CatBoost encoding", 0)
    _ = train_model(train_encoded, valid_encoded)
    print()


# Exercise 3: Feature Generation
def ex_3():
    print_("Exercise 3: Feature Generation", 0, 1)
    # -----
    # Setup
    # -----
    # Create features from   timestamps
    click_data = pd.read_csv(adtracking_file_path,
                             parse_dates=['click_time'])
    click_times = click_data['click_time']
    clicks = click_data.assign(day=click_times.dt.day.astype('uint8'),
                               hour=click_times.dt.hour.astype('uint8'),
                               minute=click_times.dt.minute.astype('uint8'),
                               second=click_times.dt.second.astype('uint8'))

    # Label encoding for categorical features
    cat_features = ['ip', 'app', 'device', 'os', 'channel']
    for feature in cat_features:
        label_encoder = preprocessing.LabelEncoder()
        clicks[feature] = label_encoder.fit_transform(clicks[feature])

    def get_data_splits(dataframe, valid_fraction=0.1):

        dataframe = dataframe.sort_values('click_time')
        valid_rows = int(len(dataframe) * valid_fraction)
        train = dataframe[:-valid_rows * 2]
        # valid size == test size, last two sections of the data
        valid = dataframe[-valid_rows * 2:-valid_rows]
        test = dataframe[-valid_rows:]

        return train, valid, test

    def train_model(train, valid, test=None, feature_cols=None):
        if feature_cols is None:
            feature_cols = train.columns.drop(['click_time', 'attributed_time',
                                               'is_attributed'])
        dtrain = lgb.Dataset(train[feature_cols], label=train['is_attributed'])
        dvalid = lgb.Dataset(valid[feature_cols], label=valid['is_attributed'])

        param = {'num_leaves': 64, 'objective': 'binary',
                 'metric': 'auc', 'seed': 7, 'verbose': -1}
        num_round = 1000
        # print("Training model. Hold on a minute to see the validation score")
        bst = lgb.train(param, dtrain, num_round, valid_sets=[dvalid],
                        early_stopping_rounds=20, verbose_eval=False)

        valid_pred = bst.predict(valid[feature_cols])
        valid_score = metrics.roc_auc_score(valid['is_attributed'], valid_pred)
        print(f"Validation AUC score: {valid_score}")

        if test is not None:
            test_pred = bst.predict(test[feature_cols])
            test_score = metrics.roc_auc_score(test['is_attributed'], test_pred)
            return bst, valid_score, test_score
        else:
            return bst, valid_score

    print_("Baseline model score", 0)
    train, valid, test = get_data_splits(clicks)
    _ = train_model(train, valid)
    print()

    # ---------------------------
    # 1. Add interaction features
    # ---------------------------
    cat_features = ['ip', 'app', 'device', 'os', 'channel']
    interactions = pd.DataFrame(index=clicks.index)

    # Iterate through each pair of features, combine them into interaction features
    encoder = LabelEncoder()
    for col1, col2 in itertools.combinations(cat_features, 2):
        new_col_name = '_'.join([col1, col2])

        # Convert to strings and combine
        new_values = clicks[col1].map(str) + "_" + clicks[col2].map(str)

        interactions[new_col_name] = encoder.fit_transform(new_values)

    clicks = clicks.join(interactions)
    print_("Score with interactions", 0)
    train, valid, test = get_data_splits(clicks)
    _ = train_model(train, valid)
    print()

    # Generating numerical features

    # -----------------------------------------
    # 2. Number of events in the past six hours
    # -----------------------------------------
    def count_past_events(series):
        series = pd.Series(series.index, index=series)
        past_events = series.rolling('6h').count() - 1
        return past_events

    ipdb.set_trace()

    # -----------------------------------
    # 3. Features from future information
    # -----------------------------------

    # ------------------------
    # 4. Time since last event
    # ------------------------
    def time_diff(series):
        """Returns a series with the time since the last timestamp in seconds."""
        return series.diff().dt.total_seconds()

    # -----------------------------------
    # 5. Number of previous app downloads
    # -----------------------------------
    def previous_attributions(series):
        """Returns a series with the number of times an app has been downloaded."""
        return series.expanding(min_periods=2).sum() - series


# Exercise 4: Feature Selection
def ex_4():
    print_("Exercise 4: Feature Selection", 0, 1)
    # ----------------------------------------
    # Which data to use for feature selection?
    # ----------------------------------------


if __name__ == '__main__':
    # ex_1()
    # ex_2()
    ex_3()
    # ex_4()
