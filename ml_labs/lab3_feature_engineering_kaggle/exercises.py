"""Kaggle - Feature Engineering (Exercises)

The TalkingData AdTracking dataset can be downloaded from
https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection
"""
import os

import ipdb
import lightgbm as lgb
import pandas as pd
from sklearn import metrics
from sklearn import preprocessing

from ml_labs.utils.genutils import print_

adtracking_file_path = os.path.expanduser('~/Data/kaggle_datasets/talkingdata_adtracking_fraud_detection/train_sample.csv')


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


if __name__ == '__main__':
    ex_1()
