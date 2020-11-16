"""Kaggle - Feature Engineering (Tutorials)

Kickstarter Projects dataset can be downloaded from
https://www.kaggle.com/kemical/kickstarter-projects
"""
import os

import lightgbm as lgb
import pandas as pd
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


if __name__ == '__main__':
    lesson_1()
