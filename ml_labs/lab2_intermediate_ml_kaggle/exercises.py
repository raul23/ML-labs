"""Kaggle - Intermediate Machine Learning (Exercises)

The Home Prices dataset can be downloaded from
https://www.kaggle.com/c/home-data-for-ml-course/data
"""
import os
from pprint import pprint

import ipdb
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from xgboost import XGBRegressor

from ml_labs.utils.genutils import print_

train_file_path = os.path.expanduser('~/Data/kaggle_datasets/home_data_for_ml_course/train.csv')
test_file_path = os.path.expanduser('~/Data/kaggle_datasets/home_data_for_ml_course/test.csv')


def load_data_for_ex_4():
    # Read the data
    X_full = pd.read_csv(train_file_path, index_col='Id')
    X_test_full = pd.read_csv(test_file_path, index_col='Id')

    # Remove rows with missing target, separate target from predictors
    X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
    y = X_full.SalePrice
    X_full.drop(['SalePrice'], axis=1, inplace=True)

    # Break off validation set from training data
    X_train_full, X_valid_full, y_train, y_valid = train_test_split(X_full, y,
                                                                    train_size=0.8, test_size=0.2,
                                                                    random_state=0)

    # "Cardinality" means the number of unique values in a column
    # Select categorical columns with relatively low cardinality (convenient but arbitrary)
    categorical_cols = [cname for cname in X_train_full.columns if
                        X_train_full[cname].nunique() < 10 and
                        X_train_full[cname].dtype == "object"]

    # Select numerical columns
    numerical_cols = [cname for cname in X_train_full.columns if
                      X_train_full[cname].dtype in ['int64', 'float64']]

    # Keep selected columns only
    my_cols = categorical_cols + numerical_cols
    X_train = X_train_full[my_cols].copy()
    X_valid = X_valid_full[my_cols].copy()
    X_test = X_test_full[my_cols].copy()

    return X_train, X_valid, y_train, y_valid, X_test, numerical_cols, categorical_cols


def load_data_for_ex_5():
    # Read the data
    train_data = pd.read_csv(train_file_path, index_col='Id')
    test_data = pd.read_csv(test_file_path, index_col='Id')

    # Remove rows with missing target, separate target from predictors
    train_data.dropna(axis=0, subset=['SalePrice'], inplace=True)
    y = train_data.SalePrice
    train_data.drop(['SalePrice'], axis=1, inplace=True)

    # Select numeric columns only
    numeric_cols = [cname for cname in train_data.columns if train_data[cname].dtype in ['int64', 'float64']]
    X = train_data[numeric_cols].copy()
    X_test = test_data[numeric_cols].copy()

    return X, y, X_test


def load_data_for_ex_6():
    # Read the data
    X = pd.read_csv(train_file_path, index_col='Id')
    X_test_full = pd.read_csv(test_file_path, index_col='Id')

    # Remove rows with missing target, separate target from predictors
    X.dropna(axis=0, subset=['SalePrice'], inplace=True)
    y = X.SalePrice
    X.drop(['SalePrice'], axis=1, inplace=True)

    # Break off validation set from training data
    X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                                    random_state=0)

    # "Cardinality" means the number of unique values in a column
    # Select categorical columns with relatively low cardinality (convenient but arbitrary)
    low_cardinality_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and
                            X_train_full[cname].dtype == "object"]

    # Select numeric columns
    numeric_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]

    # Keep selected columns only
    my_cols = low_cardinality_cols + numeric_cols
    X_train = X_train_full[my_cols].copy()
    X_valid = X_valid_full[my_cols].copy()
    X_test = X_test_full[my_cols].copy()

    # One-hot encode the data (to shorten the code, we use pandas)
    X_train = pd.get_dummies(X_train)
    X_valid = pd.get_dummies(X_valid)
    X_test = pd.get_dummies(X_test)
    X_train, X_valid = X_train.align(X_valid, join='left', axis=1)
    X_train, X_test = X_train.align(X_test, join='left', axis=1)

    return X_train, X_valid, y_train, y_valid, X_test


# Function for comparing different approaches
def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)


# Exercise 1: Introduction
def ex_1():
    print_("Exercise 1: Introduction", 0, 1)
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
    output.to_csv('ex1_submission.csv', index=False)


# Exercise 2: Missing Values
def ex_2():
    print_("Exercise 2: Missing Values", 0, 1)
    # -----
    # Setup
    # -----
    # Read the data
    X_full = pd.read_csv(train_file_path, index_col='Id')
    X_test_full = pd.read_csv(test_file_path, index_col='Id')

    # Remove rows with missing target, separate target from predictors
    X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
    y = X_full.SalePrice
    X_full.drop(['SalePrice'], axis=1, inplace=True)

    # To keep things simple, we'll use only numerical predictors
    X = X_full.select_dtypes(exclude=['object'])
    X_test = X_test_full.select_dtypes(exclude=['object'])

    # Break off validation set from training data
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                          random_state=0)

    print_("First 5 rows of the train data", 0)
    print_(X_train.head())

    # ---------------------------------
    # Step 1: Preliminary investigation
    # ---------------------------------
    print_("Shape of training data (num_rows, num_columns)", 0)
    print_(X_train.shape)

    # Number of missing values in each column of training data
    missing_val_count_by_column = (X_train.isnull().sum())
    print_("Number of missing values in each column of training data", 0)
    print_(missing_val_count_by_column[missing_val_count_by_column > 0])

    # ----------------------------------------
    # Step 2: Drop columns with missing values
    # ----------------------------------------
    # Get names of columns with missing values
    cols_with_missing = [col for col in X_train.columns
                         if X_train[col].isnull().any()]

    # Drop columns in training and validation data
    reduced_X_train = X_train.drop(cols_with_missing, axis=1)
    reduced_X_valid = X_valid.drop(cols_with_missing, axis=1)

    print_("MAE (Drop columns with missing values):", 0)
    print_(score_dataset(reduced_X_train, reduced_X_valid, y_train, y_valid))

    # ------------------
    # Step 3: Imputation
    # ------------------
    # Part A: impute missing values with the mean value along each column
    my_imputer = SimpleImputer()
    imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
    imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))

    # Imputation removed column names; put them back
    imputed_X_train.columns = X_train.columns
    imputed_X_valid.columns = X_valid.columns

    print_("MAE (Imputation):", 0)
    print_(score_dataset(imputed_X_train, imputed_X_valid, y_train, y_valid))

    # ---------------------------------
    # Step 4: Generate test predictions
    # ---------------------------------
    # Part A: use any approach of your choosing to deal with missing values
    # Preprocessed training and validation features
    # Imputation
    final_imputer = SimpleImputer(strategy='median')
    final_X_train = pd.DataFrame(final_imputer.fit_transform(X_train))
    final_X_valid = pd.DataFrame(final_imputer.transform(X_valid))

    # Imputation removed column names; put them back
    final_X_train.columns = X_train.columns
    final_X_valid.columns = X_valid.columns

    # Define and fit model
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(final_X_train, y_train)

    # Get validation predictions and MAE
    preds_valid = model.predict(final_X_valid)
    print_("MAE on valid (Your approach):", 0)
    print_(mean_absolute_error(y_valid, preds_valid))

    # Part B: preprocess your test data
    # Preprocess test data
    final_X_test = pd.DataFrame(final_imputer.transform(X_test))

    # Get test predictions
    preds_test = model.predict(final_X_test)

    # Save test predictions to file
    output = pd.DataFrame({'Id': X_test.index,
                           'SalePrice': preds_test})
    output.to_csv('ex2_submission.csv', index=False)


# Exercise 3: Categorical Variables
def ex_3():
    print_("Exercise 3: Categorical Variables", 0, 1)
    # -----
    # Setup
    # -----
    # Read the data
    X = pd.read_csv(train_file_path, index_col='Id')
    X_test = pd.read_csv(test_file_path, index_col='Id')

    # Remove rows with missing target, separate target from predictors
    X.dropna(axis=0, subset=['SalePrice'], inplace=True)
    y = X.SalePrice
    X.drop(['SalePrice'], axis=1, inplace=True)

    # To keep things simple, we'll drop columns with missing values
    cols_with_missing = [col for col in X.columns if X[col].isnull().any()]
    X.drop(cols_with_missing, axis=1, inplace=True)
    X_test.drop(cols_with_missing, axis=1, inplace=True)

    # Break off validation set from training data
    X_train, X_valid, y_train, y_valid = train_test_split(X, y,
                                                          train_size=0.8, test_size=0.2,
                                                          random_state=0)

    print_("First 5 rows from train set", 0)
    print_(X_train.head())

    # ------------------------------------------
    # Step 1: Drop columns with categorical data
    # ------------------------------------------
    # The most straightforward approach
    # Drop columns in training and validation data
    drop_X_train = X_train.select_dtypes(exclude=['object'])
    drop_X_valid = X_valid.select_dtypes(exclude=['object'])

    print_("MAE from Approach 1 (Drop categorical variables):", 0)
    print_(score_dataset(drop_X_train, drop_X_valid, y_train, y_valid))

    print("Unique values in 'Condition2' column in training data:", X_train['Condition2'].unique())
    print("\nUnique values in 'Condition2' column in validation data:", X_valid['Condition2'].unique())

    # ----------------------
    # Step 2: Label encoding
    # ----------------------
    # Part A
    # All categorical columns
    object_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]

    # Columns that can be safely label encoded
    good_label_cols = [col for col in object_cols if
                       set(X_train[col]) == set(X_valid[col])]

    # Problematic columns that will be dropped from the dataset
    bad_label_cols = list(set(object_cols) - set(good_label_cols))

    print('\nCategorical columns that will be label encoded:', good_label_cols)
    print('\nCategorical columns that will be dropped from the dataset:', bad_label_cols, end="\n\n")

    # Part B
    # Drop categorical columns that will not be encoded
    label_X_train = X_train.drop(bad_label_cols, axis=1)
    label_X_valid = X_valid.drop(bad_label_cols, axis=1)

    # Apply label encoder to the good labeled columns
    label_encoder = LabelEncoder()
    for col in good_label_cols:
        label_X_train[col] = label_encoder.fit_transform(X_train[col])
        label_X_valid[col] = label_encoder.transform(X_valid[col])

    print_("MAE from Approach 2 (Label Encoding):", 0)
    print_(score_dataset(label_X_train, label_X_valid, y_train, y_valid))

    # Get number of unique entries in each column with categorical data
    object_nunique = list(map(lambda col: X_train[col].nunique(), object_cols))
    d = dict(zip(object_cols, object_nunique))

    # Print number of unique entries by column, in ascending order
    print_("Number of unique entries by categorical column", 0)
    pprint(sorted(d.items(), key=lambda x: x[1]))
    print()

    # ---------------------------------
    # Step 3: Investigating cardinality
    # ---------------------------------
    # Part B
    # Columns that will be one-hot encoded
    low_cardinality_cols = [col for col in object_cols if X_train[col].nunique() < 10]

    # Columns that will be dropped from the dataset
    high_cardinality_cols = list(set(object_cols) - set(low_cardinality_cols))

    print('Categorical columns that will be one-hot encoded:', low_cardinality_cols)
    print('\nCategorical columns that will be dropped from the dataset:', high_cardinality_cols, end="\n\n")

    # ------------------------
    # Step 4: One-hot encoding
    # ------------------------
    # Apply one-hot encoder to each column with categorical data
    OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[low_cardinality_cols]))
    OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[low_cardinality_cols]))

    # One-hot encoding removed index; put it back
    OH_cols_train.index = X_train.index
    OH_cols_valid.index = X_valid.index

    # Remove categorical columns (will replace with one-hot encoding)
    num_X_train = X_train.drop(object_cols, axis=1)
    num_X_valid = X_valid.drop(object_cols, axis=1)

    # Add one-hot encoded columns to numerical features
    OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
    OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

    print_("MAE from Approach 3 (One-Hot Encoding):", 0)
    print_(score_dataset(OH_X_train, OH_X_valid, y_train, y_valid))


# Exercise 4: Pipelines
def ex_4():
    print_("Exercise 4: Pipelines", 0, 1)
    X_train, X_valid, y_train, y_valid, X_test, numerical_cols, categorical_cols = load_data_for_ex_4()

    print_("First 5 rows from train", 0)
    print_(X_train.head())

    # -------------------------------------
    # Preprocess the data and train a model
    # -------------------------------------
    # Preprocessing for numerical data
    numerical_transformer = SimpleImputer(strategy='constant')

    # Preprocessing for categorical data
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    # Define model
    model = RandomForestRegressor(n_estimators=100, random_state=0)

    # Bundle preprocessing and modeling code in a pipeline
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('model', model)
                          ])

    # Preprocessing of training data, fit model
    clf.fit(X_train, y_train)

    # Preprocessing of validation data, get predictions
    preds = clf.predict(X_valid)

    print('MAE:', mean_absolute_error(y_valid, preds))

    # -------------------------------
    # Step 1: Improve the performance
    # -------------------------------
    # Part A
    # Preprocessing for numerical data
    numerical_transformer = SimpleImputer(strategy='median')

    # Preprocessing for categorical data
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    model = RandomForestRegressor(n_estimators=100, random_state=0)

    # Part B
    # Bundle preprocessing and modeling code in a pipeline
    my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('model', model)
                                  ])

    # Preprocessing of training data, fit model
    my_pipeline.fit(X_train, y_train)

    # Preprocessing of validation data, get predictions
    preds = my_pipeline.predict(X_valid)

    # Evaluate the model
    score = mean_absolute_error(y_valid, preds)
    print('\nMAE:', score)

    # ---------------------------------
    # Step 2: Generate test predictions
    # ---------------------------------
    # Preprocessing of test data, fit model
    preds_test = my_pipeline.predict(X_test)

    # Save test predictions to file
    output = pd.DataFrame({'Id': X_test.index,
                           'SalePrice': preds_test})
    output.to_csv('ex4_submission.csv', index=False)


# Exercise 5: Cross-validation
def ex_5():
    print_("Exercise 5: Cross-validation", 0, 1)
    X, y, X_test = load_data_for_ex_5()

    print_("First 5 rows from X", 0)
    print_(X.head())

    my_pipeline = Pipeline(steps=[
        ('preprocessor', SimpleImputer()),
        ('model', RandomForestRegressor(n_estimators=50, random_state=0))
    ])

    # Multiply by -1 since sklearn calculates *negative* MAE
    scores = -1 * cross_val_score(my_pipeline, X, y,
                                  cv=5,
                                  scoring='neg_mean_absolute_error')

    print("Average MAE score:", scores.mean())

    # -------------------------------
    # Step 1: Write a useful function
    # -------------------------------
    def get_score(n_estimators):
        """Return the average MAE over 3 CV folds of random forest model.

        Keyword argument:
        n_estimators -- the number of trees in the forest
        """
        my_pipeline_ = Pipeline(steps=[
            ('preprocessor', SimpleImputer()),
            ('model', RandomForestRegressor(n_estimators=n_estimators, random_state=0))
        ])
        # Multiply by -1 since sklearn calculates *negative* MAE
        scores = -1 * cross_val_score(my_pipeline_, X, y,
                                      cv=3,
                                      scoring='neg_mean_absolute_error')
        # print("\nAverage MAE score (across experiments):")
        # print(scores.mean(), end="\n\n")
        return scores.mean()

    # ---------------------------------------
    # Step 2: Test different parameter values
    # ---------------------------------------
    results = dict([(i, get_score(i)) for i in range(50, 300, 50)])

    plt.plot(list(results.keys()), list(results.values()))
    plt.show()

    # -------------------------------------
    # Step 3: Find the best parameter value
    # -------------------------------------


# Exercise 6: XGBoost
def ex_6():
    print_("Exercise 6: XGBoost", 0, 1)
    X_train, X_valid, y_train, y_valid, X_test = load_data_for_ex_6()
    # -------------------
    # Step 1: Build model
    # -------------------
    # Part A
    # Define the model
    my_model_1 = XGBRegressor(random_state=0)

    # Fit the model
    my_model_1.fit(X_train, y_train)

    # Part B
    # Get predictions
    predictions_1 = my_model_1.predict(X_valid)

    # Part C
    # Calculate MAE
    mae_1 = mean_absolute_error(predictions_1, y_valid)
    print("Mean Absolute Error:", mae_1)

    # -------------------------
    # Step 2: Improve the model
    # -------------------------
    my_model_2 = XGBRegressor(random_state=0, n_estimators=1000, learning_rate=0.05, n_jobs=4)

    # Fit the model
    my_model_2.fit(X_train, y_train,
                   early_stopping_rounds=5,
                   eval_set=[(X_valid, y_valid)],
                   verbose=False)

    # Get predictions
    predictions_2 = my_model_2.predict(X_valid)

    # Calculate MAE
    mae_2 = mean_absolute_error(predictions_2, y_valid)
    print("Mean Absolute Error:" , mae_2)

    # -----------------------
    # Step 3: Break the model
    # -----------------------
    my_model_3 = XGBRegressor(random_state=0, n_estimators=10, learning_rate=0.9)

    # Fit the model
    my_model_3.fit(X_train, y_train)

    # Get predictions
    predictions_3 = my_model_3.predict(X_valid)

    # Calculate MAE
    mae_3 = mean_absolute_error(predictions_3, y_valid)
    print("Mean Absolute Error:", mae_3)


if __name__ == '__main__':
    # ex_1()
    # ex_2()
    # ex_3()
    # ex_4()
    # ex_5()
    ex_6()
    # ex_7(): no programming questions
