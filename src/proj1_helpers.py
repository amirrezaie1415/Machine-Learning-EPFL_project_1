# -*- coding: utf-8 -*-
"""some helper functions for project 1."""

# Import required packages
import csv
import numpy as np
import matplotlib.pyplot as plt


def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y == 'b')] = -1

    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1

    return y_pred


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w', newline='') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id': int(r1), 'Prediction': int(r2)})


# *** Some functions for plotting ***
# ---------------------------------------------------------------------------------------------------------------------#
def box_plot(x, feature_column):
    """ 
    Illustrating box-plot for a specific feature and printing indices of samples that can be outliers.
    """
    plt.boxplot(x[:, feature_column], showfliers=True)
    plt.title('feature column:' + str(feature_column))
    plt.show()
    index_to_delete = np.where(x[:, feature_column] == max(x[:, feature_column]))[0][0]
    print('sample (potential) outlier index:', index_to_delete)


def box_multi_plot(x):
    """
    Plotting the boxplots for all features
    """
    fig = plt.figure(figsize=(20, 15))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    for featureID in range(x.shape[1]):
        ax = fig.add_subplot(6, 5, featureID + 1)
        ax.boxplot(x[:, featureID], showfliers=True)
        ax.grid(axis='y', alpha=0.75)
        ax.set(title='Feature column:{}'.format(featureID))


def hist_multi_plot(x, color):
    """
    Plotting the histograms of each feature in a design matrix 'tx'.
    """
    fig = plt.figure(figsize=(20, 15))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    for featureID in range(x.shape[1]):
        ax = fig.add_subplot(6, 5, featureID + 1)
        ax.hist(x=x[:, featureID], bins='auto', color=color)
        ax.grid(axis='y', alpha=0.75)
        ax.set(title='Feature column:{}'.format(featureID))
    plt.show()


# *** Some functions for pre-processing ***
# ---------------------------------------------------------------------------------------------------------------------#
def divide_database(y, x, ids, PRI_jet_num):
    """ 
    Divide database into 4 categories according to "PRI_jet_num" feature values.
    
    INPUT:
    y           =       target values as a vector of the shape (N,1) in which 'N' is the number of data points.
    x           =       design matrix with shape of (N,D), where N is the number of samples and
                        D is the number of features.
    ids         =       id vector of samples
    PRI_jet_num =       the value of the feature, according to which the database is divided.
                        it can take integer values of 0, 1, 2 or 3.
    OUTPUT:
    Database of x and y corresponding to "PRI_jet_num" feature value .
    """

    # Finding the indices at which the database feature column 22, is PRI_jet_num.
    indices = np.where(x[:, 22] == PRI_jet_num)
    if len(y) != 0:
        return x[indices], y[indices], ids[indices]
    else:
        return x[indices], [], ids[indices]


def count_categorical(y, x, PRI_jet_num):
    """
    Count the number of signal and background samples with the feature "PRI_jet_num" equal to PRI_jet_num.
    
    INPUT:
    y     =         target values as a vector of the shape (N,1) in which 'N' is the number of data points.
    x     =         design matrix with shape of (N,D), where N is the number of samples and
                    D is the number of features.
    PRI_jet_num =   the value of the feature, according to which the database is divided.
                    it can take integer values of 0, 1, 2 or 3.
    OUTPUT: 
    Number of signals and background
    """
    count_signal = len(np.where(y[np.where(x[0:, 22] == PRI_jet_num)] == +1.)[0])
    count_background = len(np.where(y[np.where(x[0:, 22] == PRI_jet_num)] == -1.)[0])
    return count_signal, count_background


def cleaning_data(y, x, irrelevant_feature_columns, replace_missing_data=True):
    """
    Cleaning the database that includes
    1 - deleting features which are meaningless; and
    2 - replacing samples in which the feature DER_mass_MMC is -999 with median.
    
    INPUT:
    y           =                   target values as a vector of the shape (N,1) in which 'N' is the number of
                                    data points.
    x           =                   design matrix with shape of (N,D), where N is the number of samples and
                                    D is the number of features.
    irrelevant_feature_columns =    index of columns (features), where the values are meaningless (=-999) or constant.
    replace_missing_data       =    a flag to replace the values of the feature 'DER_mass_MMC' where it is -999 by
                                    the median of the rest of the data.
    
    OUTPUT:
    Cleaned database
    """
    # removing irrelevant feature columns
    x = np.delete(x, irrelevant_feature_columns, axis=1)
    # replace the values -999.0 in the feature vector DER_mass_MMC by the median of the data
    if replace_missing_data:
        x[np.where(x[:, 0] == -999), 0] = np.median(x[np.where(x[:, 0] != -999)[0], 0])
    return x, y


def standardize(x_train, x_test):
    """
    Z-score normalization
    
    INPUT:
    x_train     =    training design matrix, where each row is a sample and the columns are the features.
    x_test      =    test design matrix
    OUTPUT:
    Return the standardized train and test design matrices
    """
    std_train = np.std(x_train, axis=0)  # standard deviation of train features
    mean_train = np.mean(x_train, axis=0)  # mean of the train features
    standardized_train = (x_train - mean_train) / std_train  # standardize the train data
    # standardize the test data using the mean and std of the training data
    standardized_test = (x_test - mean_train) / std_train
    return standardized_train, standardized_test


def split_data(x, y, ids, ratio, seed=5):
    """
    split the data set based on the split ratio. If ratio is 0.8
    you will have 80% of your data set dedicated to training
    and the rest dedicated to validating.

    INPUT:
    y     =         target values as a rank 1 numpy array of the shape (N,) in which 'N' is the number of data points.
    x     =         design matrix with shape of (N,D), where N is the number of samples and D is the number of features.
    ids   =         ids of samples
    ratio =         Ratio between the number of training samples to total number of samples. A number between (0,1]
    seed  =         seed number for the purpose of reproducibility.
    """
    # set seed for purpose of reproducibility
    np.random.seed(seed)
    num_samples = x.shape[0]
    indices = np.arange(num_samples)
    # shuffling the indices
    np.random.shuffle(indices)
    train_indices = indices[0:int(round(x.shape[0] * ratio, 0))]
    valid_indices = indices[int(round(x.shape[0] * ratio, 0)):-1]
    x_train, y_train, ids_train = x[train_indices], y[train_indices], ids[train_indices]
    x_test, y_test, ids_test = x[valid_indices], y[valid_indices], ids[valid_indices]

    return x_train, y_train, ids_train, x_test, y_test, ids_test


def PCA(x_train, x_test):
    """
    Project the design matrix into principle directions

    INPUT:
    x_train =       test design matrix the shape of (N_train,D) where N_train is number of training samples and D is the
                    number of features.
    x_test  =       test design matrix the shape of (N_train,D) where N_train is number of training samples and D is the
                    number of features.

    OUTPUT:
    Return the projected x_train and x_test of the size (N,D)
    """
    # compute the eigenvalues and eigen-vectors of the training design matrix
    eigenvalues_train, vectors_train = np.linalg.eig(np.cov(x_train.T))
    # projected x_train into principal directions
    projected_x_train = (vectors_train.T @ x_train.T).T
    # projected x_test into principal directions
    projected_x_test = (vectors_train.T @ x_test.T).T
    return np.real(projected_x_train), np.real(projected_x_test)


# *** Functions for feature engineering. ***
# ---------------------------------------------------------------------------------------------------------------------#
def build_poly(x, degree):
    """
    Augmenting the design matrix using polynomial basis functions.

    INPUT:
    x         =         design matrix with shape of (N,D), where N is the number of samples and
                        D is the number of features.
    degree    =         order of polynomial

    OUTPUT:
    Return the augmented design matrix
    """
    augmented_x = x
    for first_degree_item in np.arange(degree + 1):
        for second_degree_item in range(0, degree - first_degree_item + 1):
            augmented_x = np.c_[augmented_x, (x ** first_degree_item) * (x ** second_degree_item)]
    return augmented_x


def build_log(x, features_with_zeros):
    """
    Augmenting the design matrix using log function.

    INPUT:
    x                    =  design matrix with shape of (N,D), where N is the number of samples and
                            D is the number of features.
    features_with_zeros  =  an array containing the index of feature columns in which there is a zero value.

    OUTPUT:
    Return the augmented design matrix

    """
    augmented_x = x
    for feature_id in range(0, x.shape[1]):
        if not (x[:, feature_id] < 0).any():
            if not (x[:, feature_id] == 0).any():
                if not (feature_id in features_with_zeros):
                    augmented_x = np.c_[augmented_x, np.log(x[:, feature_id])]
    return augmented_x


def build_log_inverse(X, features_with_zeros):
    """
    Augmenting the design matrix using inverse of log function.

    INPUT:
    X         =        design matrix the shape of (N,D) where N is number of
                       samples and D is the number of features.
    features_with_zeros  =  an array containing the index of feature columns 
                            in which there is a zero value.
    OUTPUT:
    Return the augmented design matrix

    """
    augmented_X = X
    for feature_id in range(0, X.shape[1]):
        if not (X[:, feature_id] < 0).any():
            if not (X[:, feature_id] == 0).any():
                if not (feature_id in features_with_zeros):
                    augmented_X = np.c_[augmented_X, 1/np.log(X[:, feature_id])]
    return augmented_X


def build_sqrt(X):
    """
    Augmenting the design matrix using square root function.

    INPUT:
    X         =        design matrix the shape of (N,D) where N is number of
                       samples and D is the number of features.

    OUTPUT:
    Return the augmented design matrix
    """
    augmented_X = X
    for feature_id in range(0, X.shape[1]):
        if not (X[:, feature_id] < 0).any():
            augmented_X = np.c_[augmented_X, np.sqrt(X[:, feature_id])]
    return augmented_X


def differences(X):
    """
    Augmenting the design matrix using differences between features.

    INPUT:
    X         =        design matrix the shape of (N,D) where N is number of
                       samples and D is the number of features.

    OUTPUT:
    Return the augmented design matrix
    """
    augmented_X = X
    for feature_id_1 in range(X.shape[1]):
        for feature_id_2 in range(feature_id_1, X.shape[1]):
            if (feature_id_1 != feature_id_2):
                augmented_X = np.c_[augmented_X, X[:, feature_id_1] - X[:, feature_id_2]]
    return augmented_X


def ratios(X, features_with_zeros):
    """
    Augmenting the design matrix using ratios between features.

    INPUT:
    X    =        design matrix the shape of (N,D) where 
                  N is number of samples and D is the number of features.
                                   
    features_with_zeros  =  an array containing the index of feature columns 
                            in which there is a zero value.                    
    OUTPUT:
    Return the augmented design matrix
    """
    augmented_X = X
    for feature_id_1 in range(X.shape[1]):
        for feature_id_2 in range(feature_id_1, X.shape[1]):
            if (feature_id_1 != feature_id_2):
                if not (X[:, feature_id_2] == 0).any():
                    if not (feature_id_2 in features_with_zeros):
                        augmented_X = np.c_[augmented_X, X[:, feature_id_1] / X[:, feature_id_2]]
    return augmented_X


def build_gaussian_basis(X, basis_number):
    """
    Augmenting the design matrix using gaussian basis functions.

    INPUT:
    X         =        design matrix the shape of (N,D) where N is number of
                       samples and D is the number of features.

    OUTPUT:
    Return the augmented design matrix
    """
    np.random.seed(5)
    augmented_X = X
    mu_j = np.random.uniform(-2, 2, basis_number)
    sigma_j = np.ones(basis_number)
    for feature_count in range(X.shape[1]):
        for j in range(basis_number):
            augmented_X = np.c_[augmented_X, np.exp(-(X[:, feature_count] - mu_j[j]) ** 2 / (2 * sigma_j[j] ** 2))]
    return augmented_X

# ---------------------------------------------------------------------------------------------------------------------#

def next_batch(y, tx):
    """
    Create batches of size 1, for SGD algorithm.
    
    INPUT:
    y  =          target values as a vector of the shape (N,1) in which 'N' is the number of data points.
    tx =          design matrix, where each row is a sample and the columns are the features.
                  For example, for a database containing N samples, and D features, "tx" has the shape:
                  (i) without including the
                  offset term (N,D); and with including the offset term (N,D+1).
    
    OUTPUT:
    Return a single output data and the corresponding feature vector. 
    """
    # Shuffle data
    shuffle_indices = np.random.permutation(np.arange(y.shape[0]))
    shuffled_y = y[shuffle_indices]
    shuffled_tx = tx[shuffle_indices]
    for data_index in np.arange(0, y.shape[0]):
        yield shuffled_y[[data_index]], shuffled_tx[[data_index]]


def logistic_function(z):
    """
    Compute the logistic function.
    """
    return 1 / (1 + np.exp(-z))


def predict_logistic(X, w_star):
    """"
    Compute the binary label (0 or 1)
    
    INPUT:
    X     =             design matrix, where each row is a sample and the columns are the features.
                        For example, for a database containing N samples, and D features, "X" has the shape: (i) without
                        including the offset term (N,D); and with including the offset term (N,D+1).
    w_star =            optimum weights of the shape (D,1) or (D+1,1) if the intercept is dissimilated in the weight
                        vector.
    
    OUTPUT:
    Return the prediction vector.   
    """
    pred = logistic_function(X @ w_star)
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = -1
    return pred


def distance_euclidean(X_train, X_test):
    return np.sqrt(np.sum((X_test[:, np.newaxis, :] - X_train[np.newaxis, :, :]) ** 2, axis=2))



