from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder


def preprocess_data(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pre processes data for modeling. Receives train, val and test dataframes
    and returns numpy ndarrays of cleaned up dataframes with feature engineering
    already performed.

    Arguments:
        train_df : pd.DataFrame
        val_df : pd.DataFrame
        test_df : pd.DataFrame

    Returns:
        train : np.ndarrary
        val : np.ndarrary
        test : np.ndarrary
    """
    # Print shape of input data
    print("Input train data shape: ", train_df.shape)
    print("Input val data shape: ", val_df.shape)
    print("Input test data shape: ", test_df.shape, "\n")

    # Make a copy of the dataframes
    working_train_df = train_df.copy()
    working_val_df = val_df.copy()
    working_test_df = test_df.copy()

    # 1. Correct outliers/anomalous values in numerical
    # columns (`DAYS_EMPLOYED` column).
    working_train_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)
    working_val_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)
    working_test_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)

    # 2. TODO Encode string categorical features (dytpe `object`)
    columns_types = working_train_df.dtypes
    categorical_columns = [column_name for column_name in working_train_df.columns
                if columns_types[column_name] == object]
    n_unique_values = working_train_df[categorical_columns].apply(pd.Series.nunique)

    nan_cnt_values = working_train_df[categorical_columns]\
                 .apply(pd.Series.isna).apply(pd.Series.sum)
    nan_cnt_values = pd.DataFrame(
        zip(nan_cnt_values.index, nan_cnt_values.values),
        columns=["column_name", "nan_total"])
    
    not_nan_columns = nan_cnt_values.loc[nan_cnt_values["nan_total"]==0]
    nan_columns = nan_cnt_values.loc[nan_cnt_values["nan_total"]!=0]
    
    not_nan_aux_columns = n_unique_values[not_nan_columns["column_name"].values]
    not_nan_ord_columns = not_nan_aux_columns.loc[not_nan_aux_columns == 2].index
    not_nan_cat_columns = not_nan_aux_columns.loc[not_nan_aux_columns != 2].index
    
    nan_aux_columns = n_unique_values[nan_columns["column_name"].values]
    nan_ord_columns = nan_aux_columns.loc[nan_aux_columns == 2].index
    nan_cat_columns = nan_aux_columns.loc[nan_aux_columns != 2].index
    
    n1, n2, n3 = working_train_df.shape[0], working_test_df.shape[0], working_val_df.shape[0]
    df = pd.concat([working_train_df, working_test_df, working_val_df])
    
    # OneHotEncoder and OrdinalEncoder classes are too slow, use pandas get_dummies
    # function instead
    df = pd.get_dummies(df, columns=not_nan_ord_columns, drop_first=True)
    df = pd.get_dummies(df, columns=nan_ord_columns, drop_first=True)
    df = pd.get_dummies(df, columns=not_nan_cat_columns, drop_first=False)
    df = pd.get_dummies(df, columns=nan_cat_columns, drop_first=False, dummy_na=True)
    
    working_train_encoded, working_test_encoded, working_val_encoded = df[:n1], df[n1:n1+n2], df[n1+n2:]

    # 3. TODO Impute values for all columns with missing data or, just all the columns.
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp_mean.fit(working_train_encoded)
    working_train_imputed = imp_mean.transform(working_train_encoded)
    working_test_imputed = imp_mean.transform(working_test_encoded)
    working_val_imputed = imp_mean.transform(working_val_encoded)

    # 4. TODO Feature scaling with Min-Max scaler. Apply this to all the columns.
    scaler = MinMaxScaler()
    scaler.fit(working_train_imputed)
    working_train = scaler.transform(working_train_imputed)
    working_test = scaler.transform(working_test_imputed)
    working_val = scaler.transform(working_val_imputed)

    return working_train, working_val, working_test
