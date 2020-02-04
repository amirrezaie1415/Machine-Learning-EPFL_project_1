

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from proj1_helpers import *
from implementations import *
import os


def sub_group(X, y, ids):
    """
       Create two subgroups based on the presence or non-presence (equal to -999) of feature DER_mass_MMC

       INPUT:
       y           =                   target values as a vector of the shape (N,1) in which 'N' is the number of
                                       data points.
       x           =                   design matrix with shape of (N,D), where N is the number of samples and
                                       D is the number of features.

       OUTPUT:
       Design matrix, labels and ids of two sub-groups
       """
    with_f1_index = np.where(X[:,0] == -999)[0]
    not_f1_index = np.where(X[:,0] != -999)[0]
    if len(y) != 0:
        y_not_f1,   y_with_f1   =  y[with_f1_index],   y[not_f1_index]
    X_not_f1,   X_with_f1   =  X[with_f1_index],   X[not_f1_index]
    ids_not_f1, ids_with_f1 =  ids[with_f1_index], ids[not_f1_index]
    X_not_f1 =  X_not_f1[:,1:]
    if len(y) != 0:
        return y_not_f1, X_not_f1, ids_not_f1, y_with_f1, X_with_f1, ids_with_f1 
    else:
        return [], X_not_f1, ids_not_f1, [], X_with_f1, ids_with_f1 
        


def fit(X_tr_0, y_tr_0, X_val_0, y_val_0, gamma_0,
        X_tr_1, y_tr_1, X_val_1, y_val_1, gamma_1,
        X_tr_2, y_tr_2, X_val_2, y_val_2, gamma_2,
        X_tr_3, y_tr_3, X_val_3, y_val_3, gamma_3,
        max_iters, basis_list):
    # augmentation using gaussian basis
    basis_list = basis_list
    
    # step size
    gamma_0 = gamma_0
    gamma_1 = gamma_1
    gamma_2 = gamma_2
    gamma_3 = gamma_3
    
    # maximum number of iterations
    max_iters = max_iters
    # regularization strength parameter
    lambda_ = 0
    # preallocate weights, training loss and validation loss
    w_0, w_1, w_2, w_3 = np.zeros((4, len(basis_list)), dtype = 'object')
    tr_loss_0, tr_loss_1, tr_loss_2, tr_loss_3 = np.zeros((4,len(basis_list) ))
    val_loss_0, val_loss_1, val_loss_2, val_loss_3 = np.zeros((4,len(basis_list))) 
    
    # looping over the different number of gaussian basis function
    for counter_basis, basis_number in enumerate(basis_list):
        print('[INFO]: Start learning models with augmented features using gaussian basis functions "{}"'.format(basis_number))
        
        
        # normalizing features
        X_tr_aug_0, X_val_aug_0 = standardize(X_tr_0, X_val_0)
        X_tr_aug_1, X_val_aug_1 = standardize(X_tr_1, X_val_1)
        X_tr_aug_2, X_val_aug_2 = standardize(X_tr_2, X_val_2)
        X_tr_aug_3, X_val_aug_3 = standardize(X_tr_3, X_val_3)
    
        # augmenting data using gaussian basis functions
        X_tr_aug_0, X_val_aug_0 = build_gaussian_basis(X_tr_aug_0, basis_number), build_gaussian_basis(X_val_aug_0, basis_number)
        X_tr_aug_1, X_val_aug_1 = build_gaussian_basis(X_tr_aug_1, basis_number), build_gaussian_basis(X_val_aug_1, basis_number)
        X_tr_aug_2, X_val_aug_2 = build_gaussian_basis(X_tr_aug_2, basis_number), build_gaussian_basis(X_val_aug_2, basis_number)
        X_tr_aug_3, X_val_aug_3 = build_gaussian_basis(X_tr_aug_3, basis_number), build_gaussian_basis(X_val_aug_3, basis_number)
        
        # adding the intercept term
        X_tr_aug_0, X_val_aug_0 = np.c_[np.ones(X_tr_aug_0.shape[0]),X_tr_aug_0], np.c_[np.ones(X_val_aug_0.shape[0]),X_val_aug_0]
        X_tr_aug_1, X_val_aug_1 = np.c_[np.ones(X_tr_aug_1.shape[0]),X_tr_aug_1], np.c_[np.ones(X_val_aug_1.shape[0]),X_val_aug_1]
        X_tr_aug_2, X_val_aug_2 = np.c_[np.ones(X_tr_aug_2.shape[0]),X_tr_aug_2], np.c_[np.ones(X_val_aug_2.shape[0]),X_val_aug_2]
        X_tr_aug_3, X_val_aug_3 = np.c_[np.ones(X_tr_aug_3.shape[0]),X_tr_aug_3], np.c_[np.ones(X_val_aug_3.shape[0]),X_val_aug_3]
    
        # Model learning
       
        # model for the group 0
        print('[INFO]: Start learning model for the group 0.')
        initial_w = np.zeros((X_tr_aug_0.shape[1],1),dtype = float) 
        w_0[counter_basis], tr_loss_0[counter_basis] = reg_logistic_regression(y_tr_0, X_tr_aug_0, lambda_, initial_w, max_iters, gamma_0)
        val_loss_0[counter_basis] = compute_loss_logistic(y_val_0, X_val_aug_0,w_0[counter_basis]) / X_val_aug_0.shape[0]
        y_pred_train_0 = predict_logistic(X_tr_aug_0, w_0[counter_basis])
        y_pred_train_0[y_pred_train_0 > 0] = 1
        y_pred_train_0[y_pred_train_0 < 0] = 0
        y_pred_val_0 = predict_logistic(X_val_aug_0, w_0[counter_basis])
        y_pred_val_0[y_pred_val_0 > 0] = 1
        y_pred_val_0[y_pred_val_0 < 0] = 0 
        accuracy_train = np.where(y_tr_0 - y_pred_train_0 == 0)[0].shape[0]/y_tr_0.shape[0]
        accuracy_valid = np.where(y_val_0 - y_pred_val_0 == 0)[0].shape[0]/y_val_0.shape[0]
        print('[INFO]: Accuracy_train = {:.3f}, Accuracy_val = {:.3f}'.format(accuracy_train,accuracy_valid))
        print('[INFO]: Logistic regression model has been fitted for the group 0')
        print('#######################################################################')
              
       # model for the group 1
        print('[INFO]: Start learning model for the group 1.')
        initial_w = np.zeros((X_tr_aug_1.shape[1],1),dtype = float) 
        w_1[counter_basis], tr_loss_1[counter_basis] = reg_logistic_regression(y_tr_1, X_tr_aug_1, lambda_, initial_w, max_iters, gamma_1)
        val_loss_1[counter_basis] = compute_loss_logistic(y_val_1, X_val_aug_1,w_1[counter_basis]) / X_val_aug_1.shape[0]
        y_pred_train_1 = predict_logistic(X_tr_aug_1, w_1[counter_basis])
        y_pred_train_1[y_pred_train_1 > 0] = 1
        y_pred_train_1[y_pred_train_1 < 0] = 0
        y_pred_val_1 = predict_logistic(X_val_aug_1, w_1[counter_basis])
        y_pred_val_1[y_pred_val_1 > 0] = 1
        y_pred_val_1[y_pred_val_1 < 0] = 0 
        accuracy_train = np.where(y_tr_1 - y_pred_train_1 == 0)[0].shape[0]/y_tr_1.shape[0]
        accuracy_valid = np.where(y_val_1 - y_pred_val_1 == 0)[0].shape[0]/y_val_1.shape[0]
        print('[INFO]: Accuracy_train = {:.3f}, Accuracy_val = {:.3f}'.format(accuracy_train,accuracy_valid))
        print('[INFO]: Logistic regression model has been fitted for the group 1.')
        print('#######################################################################')
        
        # model for the group 2
        print('[INFO]: Start learning model for the group 2.')
        initial_w = np.zeros((X_tr_aug_2.shape[1],1),dtype = float) 
        w_2[counter_basis], tr_loss_2[counter_basis] = reg_logistic_regression(y_tr_2, X_tr_aug_2, lambda_, initial_w, max_iters, gamma_2)
        val_loss_2[counter_basis] = compute_loss_logistic(y_val_2, X_val_aug_2,w_2[counter_basis]) / X_val_aug_2.shape[0]
        y_pred_train_2 = predict_logistic(X_tr_aug_2, w_2[counter_basis])
        y_pred_train_2[y_pred_train_2 > 0] = 1
        y_pred_train_2[y_pred_train_2 < 0] = 0
        y_pred_val_2 = predict_logistic(X_val_aug_2, w_2[counter_basis])
        y_pred_val_2[y_pred_val_2 > 0] = 1
        y_pred_val_2[y_pred_val_2 < 0] = 0 
        accuracy_train = np.where(y_tr_2 - y_pred_train_2 == 0)[0].shape[0]/y_tr_2.shape[0]
        accuracy_valid = np.where(y_val_2 - y_pred_val_2 == 0)[0].shape[0]/y_val_2.shape[0]
        print('[INFO]: Accuracy_train = {:.3f}, Accuracy_val = {:.3f}'.format(accuracy_train,accuracy_valid))
        print('[INFO]: Logistic regression model has been fitted for the group 2.')
        print('#######################################################################')
        
        # model for the group 3
        print('[INFO]: Start learning model for the group 3.')
        initial_w = np.zeros((X_tr_aug_3.shape[1],1),dtype = float) 
        w_3[counter_basis], tr_loss_3[counter_basis] = reg_logistic_regression(y_tr_3, X_tr_aug_3, lambda_, initial_w, max_iters, gamma_3)
        val_loss_3[counter_basis] = compute_loss_logistic(y_val_3, X_val_aug_3,w_3[counter_basis]) / X_val_aug_3.shape[0]
        y_pred_train_3 = predict_logistic(X_tr_aug_3, w_3[counter_basis])
        y_pred_train_3[y_pred_train_3 > 0] = 1
        y_pred_train_3[y_pred_train_3 < 0] = 0
        y_pred_val_3 = predict_logistic(X_val_aug_3, w_3[counter_basis])
        y_pred_val_3[y_pred_val_3 > 0] = 1
        y_pred_val_3[y_pred_val_3 < 0] = 0 
        accuracy_train = np.where(y_tr_3 - y_pred_train_3 == 0)[0].shape[0]/y_tr_3.shape[0]
        accuracy_valid = np.where(y_val_3 - y_pred_val_3 == 0)[0].shape[0]/y_val_3.shape[0]
        print('[INFO]: Accuracy_train = {:.3f}, Accuracy_val = {:.3f}'.format(accuracy_train,accuracy_valid))
        print('[INFO]: Logistic regression model has been fitted for the group 3.')
        print('#######################################################################')
        return  w_0, w_1, w_2, w_3