"""Kaggle - Intermediate Machine Learning (Tutorials)

The Melbourne Housing Snapshot dataset can be downloaded from
https://www.kaggle.com/glovepm/melbourne-housing
"""
import os

import ipdb
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from ml_labs.utils.genutils import print_
from exercises import score_dataset

melbourne_file_path = os.path.expanduser('~/Data/kaggle_datasets/melbourne_housing_snapshot/melb_data.csv')


# Measure Quality of Each Approach
def score_model(model, X_t, X_v, y_t, y_v):
    model.fit(X_t, y_t)
    preds = model.predict(X_v)
    return mean_absolute_error(y_v, preds)


def load_data_for_lesson_2(features=None):
    # Load data
    melbourne_data = pd.read_csv(melbourne_file_path)
    # Choose target and features
    y = melbourne_data.Price
    if features:
        X = melbourne_data[features]
    else:
        X = melbourne_data
    # Split data into training and validation data,
    # X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=0)
    return train_test_split(X, y, random_state=0)


def load_data_for_lesson_3():
    # Read the data
    data = pd.read_csv(melbourne_file_path)

    # Separate target from predictors
    y = data.Price
    X = data.drop(['Price'], axis=1)

    # Divide data into training and validation subsets
    X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                                    random_state=0)

    # Drop columns with missing values (simplest approach)
    cols_with_missing = [col for col in X_train_full.columns if X_train_full[col].isnull().any()]
    # TODO: to remove SettingWithCopyWarning
    X_train_full_copy = X_train_full.copy()
    X_valid_full_copy = X_valid_full.copy()
    del X_train_full
    del X_valid_full
    X_train_full = X_train_full_copy
    X_valid_full = X_valid_full_copy
    X_train_full.drop(cols_with_missing, axis=1, inplace=True)
    X_valid_full.drop(cols_with_missing, axis=1, inplace=True)

    # "Cardinality" means the number of unique values in a column
    # Select categorical columns with relatively low cardinality (convenient but arbitrary)
    low_cardinality_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and
                            X_train_full[cname].dtype == "object"]

    # Select numerical columns
    numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]

    # Keep selected columns only
    my_cols = low_cardinality_cols + numerical_cols
    X_train = X_train_full[my_cols].copy()
    X_valid = X_valid_full[my_cols].copy()

    return X_train, X_valid, y_train, y_valid


def load_data_for_lesson_4():
    # Read the data
    data = pd.read_csv(melbourne_file_path)

    # Separate target from predictors
    y = data.Price
    X = data.drop(['Price'], axis=1)

    # Divide data into training and validation subsets
    X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                                    random_state=0)

    # "Cardinality" means the number of unique values in a column
    # Select categorical columns with relatively low cardinality (convenient but arbitrary)
    categorical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and
                        X_train_full[cname].dtype == "object"]

    # Select numerical columns
    numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]

    # Keep selected columns only
    my_cols = categorical_cols + numerical_cols
    X_train = X_train_full[my_cols].copy()
    X_valid = X_valid_full[my_cols].copy()

    return X_train, X_valid, y_train, y_valid, numerical_cols, categorical_cols


# Lesson 2: Missing values
def lesson_2():
    print_("LESSON 2: Missing values", 0, 1)
    # ----------------------------------
    # Example: Melbourne Housing dataset
    # ----------------------------------
    # Load data
    features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea',
                'YearBuilt', 'Lattitude', 'Longtitude']
    X_train, X_valid, y_train, y_valid = load_data_for_lesson_2(features)

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


# Lesson 3: Categorical Variables
def lesson_3():
    print_("LESSON 3: Categorical Variables", 0, 1)
    # Load data
    X_train, X_valid, y_train, y_valid = load_data_for_lesson_3()

    # -------
    # Example
    # -------
    print_("First 5 rows of train data", 0)
    print_(X_train.head())

    # Get list of categorical variables
    s = (X_train.dtypes == 'object')
    object_cols = list(s[s].index)

    print_("Categorical variables:", 0)
    print_(object_cols)

    # --------------------------------------------------
    # Score from Approach 1 (Drop Categorical Variables)
    # --------------------------------------------------
    drop_X_train = X_train.select_dtypes(exclude=['object'])
    drop_X_valid = X_valid.select_dtypes(exclude=['object'])

    print_("MAE from Approach 1 (Drop categorical variables):", 0)
    print_(score_dataset(drop_X_train, drop_X_valid, y_train, y_valid))

    # --------------------------------------
    # Score from Approach 2 (Label Encoding)
    # --------------------------------------
    # Make copy to avoid changing original data
    label_X_train = X_train.copy()
    label_X_valid = X_valid.copy()

    # Apply label encoder to each column with categorical data
    label_encoder = LabelEncoder()
    for col in object_cols:
        label_X_train[col] = label_encoder.fit_transform(X_train[col])
        label_X_valid[col] = label_encoder.transform(X_valid[col])

    print_("MAE from Approach 2 (Label Encoding):", 0)
    print_(score_dataset(label_X_train, label_X_valid, y_train, y_valid))
    # We can expect an additional boost in performance if we provide
    # better-informed labels for all ordinal variables (instead of randomly
    # assigning, for each column, each unique value to a different integer,
    # like we did)

    # ----------------------------------------
    # Score from Approach 3 (One-Hot Encoding)
    # ----------------------------------------
    # Apply one-hot encoder to each column with categorical data
    OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[object_cols]))
    OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[object_cols]))

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


# Lesson 4: Pipelines
def lesson_4():
    print_("LESSON 4: Pipelines", 0, 1)
    # -------
    # Example
    # -------
    X_train, X_valid, y_train, y_valid, numerical_cols, categorical_cols = load_data_for_lesson_4()

    print_("First 5 rows from the train data", 0,)
    print_(X_train.head())

    # Build pipeline in 3 steps

    # ----------------------------------
    # Step 1: Define Preprocessing Steps
    # ----------------------------------
    # The code below:
    #
    # - imputes missing values in numerical data, and
    # - imputes missing values and applies a one-hot encoding to categorical data.

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

    # ------------------------
    # Step 2: Define the Model
    # ------------------------
    model = RandomForestRegressor(n_estimators=100, random_state=0)

    # ----------------------------------------
    # Step 3: Create and Evaluate the Pipeline
    # ----------------------------------------
    # Define a pipeline that bundles the preprocessing and modeling steps

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
    print('MAE:', score)


if __name__ == '__main__':
    # lesson_2()
    # lesson_3()
    lesson_4()
