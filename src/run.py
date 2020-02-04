# -*- coding: utf-8 -*-

# USAGE
# run.py

# 1. Import required modules

import numpy as np
from proj1_helpers import *
from proj1_helpers_run import *
from implementations import *
import os

# 2. Load the training data into feature matrix, class labels, and event ids
fileDir = os.path.dirname(os.path.abspath('run.py'))  # find the directory of the main file
parentDir = os.path.dirname(fileDir)  # go up one level
DATA_TRAIN_PATH = os.path.join(parentDir, 'data/train.csv')  # locate the training data file location
y, X, ids = load_csv_data(DATA_TRAIN_PATH)  # read training data

# Load test data
DATA_TEST_PATH = os.path.join(parentDir, 'data/test.csv')  # locate the test data file location
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)  # read test data

# change the shape of y and ids from a 1-rank array to a vector.
if y.ndim == 1:
    y = y.reshape((y.shape[0], 1))
if ids.ndim == 1:
    ids = ids.reshape((ids.shape[0], 1))
# change encoding of class labels from {-1, +1} to {0, 1}
y[y < 0] = 0
y[y > 0] = 1

# 3. Data cleaning I
# Data cleaning is performed in two main stages. In the first stage, the samples which at least one of their features
# is detected as outlier (using boxplot + visual inspection) are removed.

outlier_indices = [202795, 191362, 182617, 173649, 171360, 153775, 151832, 150377,
                   132402, 118381, 117359, 84175, 75797, 68117,  52754, 49297, 19843, 7343]
X = np.delete(X, outlier_indices, axis=0)  # removing outliers
y = np.delete(y, outlier_indices, axis=0)  # removing outliers
ids = np.delete(ids, outlier_indices, axis=0)  # removing outliers

# initiation of the list for label prediction of test data of all models
y_pred_ids_con_list = []

# loop to build a number of various models from a fraction of training data to create variance amongst the data by
# smapling and replacing data
bagging_number = 301
for i in range(bagging_number):
    print('Model: {}'.format(i))
    # 4. Splitting data into validation and train sets
    ratio = 0.1
    X_tr, y_tr, ids_tr, X_val, y_val, ids_val = split_data(X, y, ids, ratio, seed=range(0, 1000)[i])

    # 5. Categorize data into 4 groups
    # The data is categorized into 4 groups, based on the categorical feature "PRI_jet_num",
    # which can take a value from {0, 1, 2 ,3}.
    
    # ## 5.1 Training data ## #
    X_tr_0, y_tr_0, ids_tr_0 = divide_database(y_tr, X_tr, ids_tr, PRI_jet_num=0)
    X_tr_1, y_tr_1, ids_tr_1 = divide_database(y_tr, X_tr, ids_tr, PRI_jet_num=1)
    X_tr_2, y_tr_2, ids_tr_2 = divide_database(y_tr, X_tr, ids_tr, PRI_jet_num=2)
    X_tr_3, y_tr_3, ids_tr_3 = divide_database(y_tr, X_tr, ids_tr, PRI_jet_num=3)
    
    # ## 5.2 Validation data ## #
    X_val_0, y_val_0, ids_val_0 = divide_database(y_val, X_val, ids_val, PRI_jet_num=0)
    X_val_1, y_val_1, ids_val_1 = divide_database(y_val, X_val, ids_val, PRI_jet_num=1)
    X_val_2, y_val_2, ids_val_2 = divide_database(y_val, X_val, ids_val, PRI_jet_num=2)
    X_val_3, y_val_3, ids_val_3 = divide_database(y_val, X_val, ids_val, PRI_jet_num=3)
    
    # 6. Data cleaning II
    # For each group of data, there are some features that are constant or meaningless (equal to -999);
    # these features are removed from the design matrix of each category (group).
    # Also some features are highly correlated. This can lead to over-bias of the results to these features.
    # To avoid this issue, correlated features are removed from the data.
    
    # ## 6.1 Training data ## #
    X_tr_0, y_tr_0 = cleaning_data(y_tr_0, X_tr_0, irrelevant_feature_columns=[4, 5, 6, 12, 22, 23, 24, 25, 26, 27, 28,
                                                                               29], replace_missing_data=False)
    X_tr_1, y_tr_1 = cleaning_data(y_tr_1, X_tr_1, irrelevant_feature_columns=[4, 5, 6, 12, 22, 26, 27, 28],
                                   replace_missing_data=False)
    X_tr_2, y_tr_2 = cleaning_data(y_tr_2, X_tr_2, irrelevant_feature_columns=[22], replace_missing_data=False)
    X_tr_3, y_tr_3 = cleaning_data(y_tr_3, X_tr_3, irrelevant_feature_columns=[22], replace_missing_data=False)
    
    # ## 6.2 Validation data ## #
    X_val_0, y_val_0 = cleaning_data(y_val_0, X_val_0, irrelevant_feature_columns=[4, 5, 6, 12, 22, 23, 24, 25, 26, 27,
                                                                                   28, 29], replace_missing_data=False)
    X_val_1, y_val_1 = cleaning_data(y_val_1, X_val_1, irrelevant_feature_columns=[4, 5, 6, 12, 22, 26, 27, 28],
                                     replace_missing_data=False)
    X_val_2, y_val_2 = cleaning_data(y_val_2, X_val_2, irrelevant_feature_columns=[22], replace_missing_data=False)
    X_val_3, y_val_3 = cleaning_data(y_val_3, X_val_3, irrelevant_feature_columns=[22], replace_missing_data=False)

    # removing features that have high linear correlation 
    X_val_0, _ = cleaning_data(y_val_0, X_val_0, [3], replace_missing_data=False)
    X_val_1, _ = cleaning_data(y_val_1, X_val_1, [18], replace_missing_data=False)
    X_tr_0, _ = cleaning_data(y_tr_0, X_tr_0, [3], replace_missing_data=False)
    X_tr_1, _ = cleaning_data(y_tr_1, X_tr_1, [18], replace_missing_data=False)

    # Some of the values recorded for the feature "DER_mass_MMC" are meaningless (equal to -999),
    # therefore, based on the presence or non-presence of this feature, the data is divided into subgroups.
    # EXAMPLE 1 : X_tr_0_not_f1: The part of the data in which the feature "PRI_jet_num" is equal to 0 while the
    # feature "DER_mass_MMC" is equal to -999.
    # EXAMPLE 3 : X_tr_3_not_f1: The part of the data in which the feature "PRI_jet_num" is equal to 3 while the
    # feature "DER_mass_MMC" is equal to -999.
    # Training Data
    y_tr_0_not_f1, X_tr_0_not_f1, ids_tr_0_not_f1, y_tr_0_with_f1, X_tr_0_with_f1, ids_tr_0_with_f1 = sub_group(
        X_tr_0, y_tr_0, ids_tr_0)
    y_tr_1_not_f1, X_tr_1_not_f1, ids_tr_1_not_f1, y_tr_1_with_f1, X_tr_1_with_f1, ids_tr_1_with_f1 = sub_group(
        X_tr_1, y_tr_1, ids_tr_1)
    y_tr_2_not_f1, X_tr_2_not_f1, ids_tr_2_not_f1, y_tr_2_with_f1, X_tr_2_with_f1, ids_tr_2_with_f1 = sub_group(
        X_tr_2, y_tr_2, ids_tr_2)
    y_tr_3_not_f1, X_tr_3_not_f1, ids_tr_3_not_f1, y_tr_3_with_f1, X_tr_3_with_f1, ids_tr_3_with_f1 = sub_group(
        X_tr_3, y_tr_3, ids_tr_3)
    
    # Validation Data
    y_val_0_not_f1, X_val_0_not_f1, ids_val_0_not_f1, y_val_0_with_f1, X_val_0_with_f1, ids_val_0_with_f1 = sub_group(
        X_val_0, y_val_0, ids_val_0)
    y_val_1_not_f1, X_val_1_not_f1, ids_val_1_not_f1, y_val_1_with_f1, X_val_1_with_f1, ids_val_1_with_f1 = sub_group(
        X_val_1, y_val_1, ids_val_1)
    y_val_2_not_f1, X_val_2_not_f1, ids_val_2_not_f1, y_val_2_with_f1, X_val_2_with_f1, ids_val_2_with_f1 = sub_group(
        X_val_2, y_val_2, ids_val_2)
    y_val_3_not_f1, X_val_3_not_f1, ids_val_3_not_f1, y_val_3_with_f1, X_val_3_with_f1, ids_val_3_with_f1 = sub_group(
        X_val_3, y_val_3, ids_val_3)

    # 7. Learning models using gaussian basis function for augmenting data
    # learning model for subgroup not f1
    w_0_not_f1, w_1_not_f1, w_2_not_f1, w_3_not_f1 = fit(X_tr_0_not_f1, y_tr_0_not_f1, X_val_0_not_f1,
                                                         y_val_0_not_f1, 10 ** -6 * 5,
                                                         X_tr_1_not_f1, y_tr_1_not_f1, X_val_1_not_f1,
                                                         y_val_1_not_f1, 10 ** -6 * 5,
                                                         X_tr_2_not_f1, y_tr_2_not_f1, X_val_2_not_f1,
                                                         y_val_2_not_f1, 10 ** -6 * 5,
                                                         X_tr_3_not_f1, y_tr_3_not_f1, X_val_3_not_f1,
                                                         y_val_3_not_f1, 10 ** -6 * 2,
                                                         max_iters=10000, basis_list=[15])

    # learning model for subgroup with f1
    w_0_with_f1, w_1_with_f1, w_2_with_f1, w_3_with_f1 = fit(X_tr_0_with_f1, y_tr_0_with_f1, X_val_0_with_f1,
                                                             y_val_0_with_f1, 10 ** -6 * 20,
                                                             X_tr_1_with_f1, y_tr_1_with_f1, X_val_1_with_f1,
                                                             y_val_1_with_f1, 10 ** -6 * 20,
                                                             X_tr_2_with_f1, y_tr_2_with_f1, X_val_2_with_f1,
                                                             y_val_2_with_f1, 10 ** -6 * 35,
                                                             X_tr_3_with_f1, y_tr_3_with_f1, X_val_3_with_f1,
                                                             y_val_3_with_f1, 10 ** -6 * 60,
                                                             max_iters=20000, basis_list=[10])

    # 8. Prediction of targets for test data
    
    # Divide the test data corresponding to the feature "PRI_jet_num"
    X_test_0, _, ids_test_0 = divide_database([], tX_test, ids_test, PRI_jet_num=0)
    X_test_1, _, ids_test_1 = divide_database([], tX_test, ids_test, PRI_jet_num=1)
    X_test_2, _, ids_test_2 = divide_database([], tX_test, ids_test, PRI_jet_num=2)
    X_test_3, _, ids_test_3 = divide_database([], tX_test, ids_test, PRI_jet_num=3)

    # Change the shape of ids from a 1-rank array to a vector.
    ids_test_0 = ids_test_0[:, np.newaxis]
    ids_test_1 = ids_test_1[:, np.newaxis]
    ids_test_2 = ids_test_2[:, np.newaxis]
    ids_test_3 = ids_test_3[:, np.newaxis]
    
    # cleaning test data (Removing features as per training data)
    X_test_0, _ = cleaning_data([], X_test_0, irrelevant_feature_columns=[4, 5, 6, 12, 22, 23, 24, 25, 26, 27, 28, 29],
                                replace_missing_data=False)
    X_test_1, _ = cleaning_data([], X_test_1, irrelevant_feature_columns=[4, 5, 6, 12, 22, 26, 27, 28],
                                replace_missing_data=False)
    X_test_2, _ = cleaning_data([], X_test_2, irrelevant_feature_columns=[22], replace_missing_data=False)
    X_test_3, _ = cleaning_data([], X_test_3, irrelevant_feature_columns=[22], replace_missing_data=False)

    # removing features that have high linear correlation (as per training data)
    X_test_0, _ = cleaning_data([], X_test_0, [3], replace_missing_data=False)
    X_test_1, _ = cleaning_data([], X_test_1, [18], replace_missing_data=False)
    
    _, X_test_0_not_f1, ids_test_0_not_f1, _, X_test_0_with_f1, ids_test_0_with_f1 = sub_group(X_test_0, [], ids_test_0)
    _, X_test_1_not_f1, ids_test_1_not_f1, _, X_test_1_with_f1, ids_test_1_with_f1 = sub_group(X_test_1, [], ids_test_1)
    _, X_test_2_not_f1, ids_test_2_not_f1, _, X_test_2_with_f1, ids_test_2_with_f1 = sub_group(X_test_2, [], ids_test_2)
    _, X_test_3_not_f1, ids_test_3_not_f1, _, X_test_3_with_f1, ids_test_3_with_f1 = sub_group(X_test_3, [], ids_test_3)

    # 8.1 Prediction of target for test data for subgroup 'not_f1'
    # test set is standardized (z-score normalization).
    _, X_test_aug_not_f1_0 = standardize(X_tr_0_not_f1, X_test_0_not_f1)
    _, X_test_aug_not_f1_1 = standardize(X_tr_1_not_f1, X_test_1_not_f1)
    _, X_test_aug_not_f1_2 = standardize(X_tr_2_not_f1, X_test_2_not_f1)
    _, X_test_aug_not_f1_3 = standardize(X_tr_3_not_f1, X_test_3_not_f1)
    
    # augmenting data features with Gaussian basis functions
    X_test_aug_not_f1_0 = build_gaussian_basis(X_test_aug_not_f1_0, 15)
    X_test_aug_not_f1_1 = build_gaussian_basis(X_test_aug_not_f1_1, 15)
    X_test_aug_not_f1_2 = build_gaussian_basis(X_test_aug_not_f1_2, 15)
    X_test_aug_not_f1_3 = build_gaussian_basis(X_test_aug_not_f1_3, 15)

    # adding the intercept term
    X_test_aug_not_f1_0 = np.c_[np.ones(X_test_aug_not_f1_0.shape[0]), X_test_aug_not_f1_0]
    X_test_aug_not_f1_1 = np.c_[np.ones(X_test_aug_not_f1_1.shape[0]), X_test_aug_not_f1_1]
    X_test_aug_not_f1_2 = np.c_[np.ones(X_test_aug_not_f1_2.shape[0]), X_test_aug_not_f1_2]
    X_test_aug_not_f1_3 = np.c_[np.ones(X_test_aug_not_f1_3.shape[0]), X_test_aug_not_f1_3]

    # Class prediction
    y_test_pred_not_f1_0 = predict_logistic(X_test_aug_not_f1_0, w_0_not_f1[0])
    y_test_pred_not_f1_1 = predict_logistic(X_test_aug_not_f1_1, w_1_not_f1[0])
    y_test_pred_not_f1_2 = predict_logistic(X_test_aug_not_f1_2, w_2_not_f1[0])
    y_test_pred_not_f1_3 = predict_logistic(X_test_aug_not_f1_3, w_3_not_f1[0])
    
    # Concatenating ids and predicted targets
    ids_con_not_f1_0 = np.concatenate((ids_test_0_not_f1, ids_test_1_not_f1, ids_test_2_not_f1, ids_test_3_not_f1))
    y_pred_con_not_f1_0 = np.concatenate((y_test_pred_not_f1_0, y_test_pred_not_f1_1, y_test_pred_not_f1_2,
                                          y_test_pred_not_f1_3))

    # 8.2 Prediction of target for test data for subgroup 'with_f1'
    # test set is standardized (z-score normalization).
    _, X_test_aug_with_f1_0 = standardize(X_tr_0_with_f1, X_test_0_with_f1)
    _, X_test_aug_with_f1_1 = standardize(X_tr_1_with_f1, X_test_1_with_f1)
    _, X_test_aug_with_f1_2 = standardize(X_tr_2_with_f1, X_test_2_with_f1)
    _, X_test_aug_with_f1_3 = standardize(X_tr_3_with_f1, X_test_3_with_f1)
    
    # augmenting data features with Gaussian basis functions
    X_test_aug_with_f1_0 = build_gaussian_basis(X_test_aug_with_f1_0, 10)
    X_test_aug_with_f1_1 = build_gaussian_basis(X_test_aug_with_f1_1, 10)
    X_test_aug_with_f1_2 = build_gaussian_basis(X_test_aug_with_f1_2, 10)
    X_test_aug_with_f1_3 = build_gaussian_basis(X_test_aug_with_f1_3, 10)

    # adding the intercept term
    X_test_aug_with_f1_0 = np.c_[np.ones(X_test_aug_with_f1_0.shape[0]),X_test_aug_with_f1_0]
    X_test_aug_with_f1_1 = np.c_[np.ones(X_test_aug_with_f1_1.shape[0]),X_test_aug_with_f1_1]
    X_test_aug_with_f1_2 = np.c_[np.ones(X_test_aug_with_f1_2.shape[0]),X_test_aug_with_f1_2]
    X_test_aug_with_f1_3 = np.c_[np.ones(X_test_aug_with_f1_3.shape[0]),X_test_aug_with_f1_3]

    # Class prediction
    y_test_pred_with_f1_0 = predict_logistic(X_test_aug_with_f1_0, w_0_with_f1[0])
    y_test_pred_with_f1_1 = predict_logistic(X_test_aug_with_f1_1, w_1_with_f1[0])
    y_test_pred_with_f1_2 = predict_logistic(X_test_aug_with_f1_2, w_2_with_f1[0])
    y_test_pred_with_f1_3 = predict_logistic(X_test_aug_with_f1_3, w_3_with_f1[0])

    # Concatenating ids and predicted targets
    ids_con_with_f1_0 = np.concatenate((ids_test_0_with_f1, ids_test_1_with_f1, ids_test_2_with_f1, ids_test_3_with_f1))
    y_pred_con_with_f1_0 = np.concatenate((y_test_pred_with_f1_0, y_test_pred_with_f1_1, y_test_pred_with_f1_2,
                                           y_test_pred_with_f1_3))

    # 8.3 Concatenating the results ('not_f1' and 'with_f1' subgroups)
    ids_con = np.concatenate((ids_con_not_f1_0, ids_con_with_f1_0))  # Concatenating ids
    y_pred_con = np.concatenate((y_pred_con_not_f1_0, y_pred_con_with_f1_0))  # Concatenating target predictions
    y_pred_ids_con = np.concatenate((y_pred_con, ids_con), axis=1)  # Concatenating target predictions and ids
    y_pred_ids_con = y_pred_ids_con[y_pred_ids_con[:, 1].argsort()]  # sort the target predictions according to ids
    y_pred_ids_con = y_pred_ids_con.astype('int')  # set data type as integer
    y_pred_ids_con_list.append(y_pred_ids_con)  # store predicted targets for the model in a list

# counting the votes for each id
voting = 0
for bagging_count in range(bagging_number):
    voting += y_pred_ids_con_list[bagging_count]
    
# Applying the majority voting for selecting the class labels.
# as the target predictions are either 1 or -1, positive values reflect majority voting for 1 s and and vice versa.
majority_voting = voting[:, 0]  # just consider the predicted targets
majority_voting[majority_voting > 0] = 1
majority_voting[majority_voting < 0] = -1

# creating output file
OUTPUT_PATH = os.path.join(parentDir, 'data/submission_best.csv')
create_csv_submission(ids_test, majority_voting, OUTPUT_PATH)


