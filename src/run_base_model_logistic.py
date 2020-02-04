# 1. Import required modules
import numpy as np
from proj1_helpers import *
from implementations import *
import os

# 2. Load the training data into feature matrix, class labels, and event ids
fileDir = os.path.dirname(os.path.abspath('run_base_model_logistic.py'))
parentDir = os.path.dirname(fileDir)
DATA_TRAIN_PATH = os.path.join(parentDir, 'data/train.csv')
y, X, ids = load_csv_data(DATA_TRAIN_PATH)

# change the shape of y and ids from a 1 rank array to a vector.
if y.ndim == 1:
    y = y.reshape((y.shape[0], 1))
if ids.ndim == 1:
    ids = ids.reshape((ids.shape[0], 1))

# change encoding class labels from {-1, +1} to {0, 1}
y[y < 0] = 0
y[y > 0] = 1

# 3. Data cleaning I
    
# Data cleaning is performed in two main stages. In the first stage, the samples where
# at least one of their features is detected as an outlier (using boxplot + visual inspection)
# are removed.

outlier_indices = [202795, 191362, 182617, 173649, 
                   171360, 153775, 151832, 150377, 
                   132402, 118381, 117359, 84175, 
                   75797, 68117,  52754, 49297, 19843, 7343]
X = np.delete(X, outlier_indices, axis=0)
y = np.delete(y, outlier_indices, axis=0)
ids = np.delete(ids, outlier_indices, axis=0)


# 4. Splitting data into validation and train sets

ratio = 0.8
X_tr, y_tr, ids_tr, X_val, y_val,ids_val = split_data(X, y,ids,ratio, seed=5)

# 5. Categorize data into 4 groups 

# The data is categorized into 4 groups based on the categorical feature "PRI_jet_num",
# which can take a value from {0, 1, 2 ,3}.

    # 5.1 Training data
X_tr_0, y_tr_0, _ = divide_database(y_tr, X_tr,ids_tr,PRI_jet_num = 0)
X_tr_1, y_tr_1, _ = divide_database(y_tr, X_tr,ids_tr,PRI_jet_num = 1)
X_tr_2, y_tr_2, _ = divide_database(y_tr, X_tr,ids_tr,PRI_jet_num = 2)
X_tr_3, y_tr_3, _ = divide_database(y_tr, X_tr,ids_tr,PRI_jet_num = 3)

    # 5.2 Validation data
X_val_0, y_val_0, ids_val_0 = divide_database(y_val, X_val,ids_val,PRI_jet_num = 0)
X_val_1, y_val_1, ids_val_1 = divide_database(y_val, X_val,ids_val,PRI_jet_num = 1)
X_val_2, y_val_2, ids_val_2 = divide_database(y_val, X_val,ids_val,PRI_jet_num = 2)
X_val_3, y_val_3, ids_val_3 = divide_database(y_val, X_val,ids_val,PRI_jet_num = 3)

# 6. Data cleaning II

# For each group of data, there are some features that are constant or meaningless (equal to -999);
# these features are removed from the design matrix of each category (group).
# Moreover, some of the values recorded for the feature "DER_mass_MMC" are meaningless (equal to -999),
# these values are 'replaced' with the median the feature "DER_mass_MMC"  of each group.

     # 6.1 Training data
X_tr_0,y_tr_0 = cleaning_data(y_tr_0, X_tr_0, irrelevant_feature_columns=[4,5,6,12,22,23,24,25,26,27,28,29], replace_missing_data = True)
X_tr_1,y_tr_1 = cleaning_data(y_tr_1, X_tr_1, irrelevant_feature_columns=[4,5,6,12,22,26,27,28], replace_missing_data = True)
X_tr_2,y_tr_2 = cleaning_data(y_tr_2, X_tr_2, irrelevant_feature_columns=[22], replace_missing_data = True)
X_tr_3,y_tr_3 = cleaning_data(y_tr_3, X_tr_3, irrelevant_feature_columns=[22], replace_missing_data = True)

    # 6.2 Validation data
X_val_0,y_val_0 = cleaning_data(y_val_0, X_val_0, irrelevant_feature_columns=[4,5,6,12,22,23,24,25,26,27,28,29], replace_missing_data = True)
X_val_1,y_val_1 = cleaning_data(y_val_1, X_val_1, irrelevant_feature_columns=[4,5,6,12,22,26,27,28], replace_missing_data = True)
X_val_2,y_val_2 = cleaning_data(y_val_2, X_val_2, irrelevant_feature_columns=[22], replace_missing_data = True)
X_val_3,y_val_3 = cleaning_data(y_val_3, X_val_3, irrelevant_feature_columns=[22], replace_missing_data = True)


# remove features with high respective correlations
X_val_0,_ = cleaning_data(y_val_0, X_val_0, [3], replace_missing_data = True)
X_val_1,_ = cleaning_data(y_val_1, X_val_1, [18], replace_missing_data = True)
X_tr_0,_ = cleaning_data(y_tr_0, X_tr_0, [3], replace_missing_data = False)
X_tr_1,_ = cleaning_data(y_tr_1, X_tr_1, [18], replace_missing_data = False)
    

# 7. Feature Conditioning + Learning algorithm (regularized logistic regression using gradient descent algorithm)


# step size
gamma = 2 * 10 ** -6
# maximum number of iterations
max_iters = 5000
# regularization strength parameter
lambda_list = [0, 10 ** -10, 10 ** -6, 10 ** 2, 10 ** 4]



# Both train and validation sets are standardized (z-score normalization).
# NOTICE: The mean and variance of the training data is used to normalize the validation data. 
X_tr_stan_0, X_val_stan_0 = standardize(X_tr_0, X_val_0)
X_tr_stan_1, X_val_stan_1 = standardize(X_tr_1, X_val_1)
X_tr_stan_2, X_val_stan_2 = standardize(X_tr_2, X_val_2)
X_tr_stan_3, X_val_stan_3 = standardize(X_tr_3, X_val_3)

# adding the intercept term
X_tr_stan_0, X_val_stan_0 = np.c_[np.ones(X_tr_stan_0.shape[0]),X_tr_stan_0], np.c_[np.ones(X_val_stan_0.shape[0]),X_val_stan_0]
X_tr_stan_1, X_val_stan_1 = np.c_[np.ones(X_tr_stan_1.shape[0]),X_tr_stan_1], np.c_[np.ones(X_val_stan_1.shape[0]),X_val_stan_1]
X_tr_stan_2, X_val_stan_2 = np.c_[np.ones(X_tr_stan_2.shape[0]),X_tr_stan_2], np.c_[np.ones(X_val_stan_2.shape[0]),X_val_stan_2]
X_tr_stan_3, X_val_stan_3 = np.c_[np.ones(X_tr_stan_3.shape[0]),X_tr_stan_3], np.c_[np.ones(X_val_stan_3.shape[0]),X_val_stan_3]


# preallocate weights, training loss and validation loss
w_0, w_1, w_2, w_3 = np.zeros((4, len(lambda_list)), dtype='object')
tr_loss_0, tr_loss_1, tr_loss_2, tr_loss_3 = np.zeros((4, len(lambda_list)))
val_loss_0, val_loss_1, val_loss_2, val_loss_3 = np.zeros((4, len(lambda_list)))
accuracy_train_0, accuracy_train_1, accuracy_train_2, accuracy_train_3 = np.zeros((4, len(lambda_list)))
accuracy_valid_0, accuracy_valid_1, accuracy_valid_2, accuracy_valid_3 = np.zeros((4, len(lambda_list)))

# looping over the different lambdas
for counter_lambda, lambda_ in enumerate(lambda_list):
   
    # Model learning
    # model for the group 0
    print('#######################################################################')
    print('[INFO]: Start learning model for the group 0. (lambda = {:.1e})'.format(lambda_))
    # weight initialization
    initial_w = np.zeros((X_tr_stan_0.shape[1],1),dtype = float) 
    w_0[counter_lambda], tr_loss_0[counter_lambda] = reg_logistic_regression(y_tr_0, X_tr_stan_0, lambda_, initial_w, max_iters, gamma)
    val_loss_0[counter_lambda] = compute_loss_logistic(y_val_0, X_val_stan_0, w_0[counter_lambda])
    y_pred_train_0 = predict_logistic(X_tr_stan_0, w_0[counter_lambda])
    # map to {0, 1}
    y_pred_train_0[y_pred_train_0 > 0] = 1
    y_pred_train_0[y_pred_train_0 < 0] = 0
    
    y_pred_val_0 = predict_logistic(X_val_stan_0, w_0[counter_lambda])
    # map to {0, 1}
    y_pred_val_0[y_pred_val_0 > 0] = 1
    y_pred_val_0[y_pred_val_0 < 0] = 0 
    accuracy_train_0[counter_lambda] = np.where(y_tr_0 - y_pred_train_0 == 0)[0].shape[0]/y_tr_0.shape[0]
    accuracy_valid_0[counter_lambda] = np.where(y_val_0 - y_pred_val_0 == 0)[0].shape[0]/y_val_0.shape[0]
    print('[INFO]: Accuracy_train = {:.3f}, Accuracy_val = {:.3f}'.format(accuracy_train_0[counter_lambda],accuracy_valid_0[counter_lambda]))
    print('[INFO]: Logistic regression model has been fitted for the group 0.')
    print('#######################################################################')
          
    # model for the group 1
    print('[INFO]: Start learning model for the group 1. (lambda = {:.1e})'.format(lambda_))
    initial_w = np.zeros((X_tr_stan_1.shape[1], 1), dtype=float)
    w_1[counter_lambda], tr_loss_1[counter_lambda] = reg_logistic_regression(y_tr_1, X_tr_stan_1, lambda_, initial_w, max_iters, gamma)
    val_loss_1[counter_lambda] = compute_loss_logistic(y_val_1, X_val_stan_1, w_1[counter_lambda])
    y_pred_train_1 = predict_logistic(X_tr_stan_1, w_1[counter_lambda])
    # map to {0, 1}
    y_pred_train_1[y_pred_train_1 > 0] = 1
    y_pred_train_1[y_pred_train_1 < 0] = 0

    y_pred_val_1 = predict_logistic(X_val_stan_1, w_1[counter_lambda])
    # map to {0, 1}
    y_pred_val_1[y_pred_val_1 > 0] = 1
    y_pred_val_1[y_pred_val_1 < 0] = 0 
    accuracy_train_1[counter_lambda] = np.where(y_tr_1 - y_pred_train_1 == 0)[0].shape[0]/y_tr_1.shape[0]
    accuracy_valid_1[counter_lambda] = np.where(y_val_1 - y_pred_val_1 == 0)[0].shape[0]/y_val_1.shape[0]
    print('[INFO]: Accuracy_train = {:.3f}, Accuracy_val = {:.3f}'.format(accuracy_train_1[counter_lambda],accuracy_valid_1[counter_lambda]))
    print('[INFO]: Logistic regression model has been fitted for the group 1')
    print('#######################################################################')
    
    # model for the group 2
    print('[INFO]: Start learning model for the group 2. (lambda = {:.1e})'.format(lambda_))
    initial_w = np.zeros((X_tr_stan_2.shape[1],1),dtype = float) 
    w_2[counter_lambda], tr_loss_2[counter_lambda] = reg_logistic_regression(y_tr_2, X_tr_stan_2, lambda_, initial_w, max_iters, gamma)
    val_loss_2[counter_lambda] = compute_loss_logistic(y_val_2, X_val_stan_2, w_2[counter_lambda])
    y_pred_train_2 = predict_logistic(X_tr_stan_2, w_2[counter_lambda])
    # map to {0, 1}
    y_pred_train_2[y_pred_train_2 > 0] = 1
    y_pred_train_2[y_pred_train_2 < 0] = 0
    
    y_pred_val_2 = predict_logistic(X_val_stan_2, w_2[counter_lambda])
    # map to {0, 1}
    y_pred_val_2[y_pred_val_2 > 0] = 1
    y_pred_val_2[y_pred_val_2 < 0] = 0 
    accuracy_train_2[counter_lambda] = np.where(y_tr_2 - y_pred_train_2 == 0)[0].shape[0]/y_tr_2.shape[0]
    accuracy_valid_2[counter_lambda] = np.where(y_val_2 - y_pred_val_2 == 0)[0].shape[0]/y_val_2.shape[0]
    print('[INFO]: Accuracy_train = {:.3f}, Accuracy_val = {:.3f}'.format(accuracy_train_2[counter_lambda], accuracy_valid_2[counter_lambda]))
    
    print('[INFO]: Logistic regression model has been fitted for the group 2.')
    print('#######################################################################')
    
    # model for the group 3
    print('[INFO]: Start learning model for the group 3. (lambda = {:.1e})'.format(lambda_))
    initial_w = np.zeros((X_tr_stan_3.shape[1],1),dtype = float) 
    w_3[counter_lambda], tr_loss_3[counter_lambda] = reg_logistic_regression(y_tr_3, X_tr_stan_3, lambda_, initial_w, max_iters, gamma)
    val_loss_3[counter_lambda] = compute_loss_logistic(y_val_3, X_val_stan_3, w_3[counter_lambda])
    y_pred_train_3 = predict_logistic(X_tr_stan_3, w_3[counter_lambda])
    # map to {0, 1}
    y_pred_train_3[y_pred_train_3 > 0] = 1
    y_pred_train_3[y_pred_train_3 < 0] = 0
    
    y_pred_val_3 = predict_logistic(X_val_stan_3, w_3[counter_lambda])
    # map to {0, 1}
    y_pred_val_3[y_pred_val_3 > 0] = 1
    y_pred_val_3[y_pred_val_3 < 0] = 0 
    accuracy_train_3[counter_lambda] = np.where(y_tr_3 - y_pred_train_3 == 0)[0].shape[0]/y_tr_3.shape[0]
    accuracy_valid_3[counter_lambda] = np.where(y_val_3 - y_pred_val_3 == 0)[0].shape[0]/y_val_3.shape[0]
    print('[INFO]: Accuracy_train = {:.3f}, Accuracy_val = {:.3f}'.format(accuracy_train_3[counter_lambda], accuracy_valid_3[counter_lambda]))
    print('[INFO]: Logistic regression model has been fitted for the group 3.')
    print('#######################################################################')

# write the accuracies into a text file
f = open("accuracy.txt", "a")
f.write('##################################################\n\n')
f.write('***Base Model***\n')
for counter_lambda, lambda_ in enumerate(lambda_list):
    f.write('\t%lambda:' + str(lambda_) + '\n')
    f.write('\t\t-Train Accuracy:    #Group0:' + '{:.3f}'.format(accuracy_train_0[counter_lambda]) + '    #Group1:' + '{:.3f}'.format(accuracy_train_1[counter_lambda]) + '    #Group2:' + '{:.3f}'.format(accuracy_train_2[counter_lambda]) + '    #Group3:' + '{:.3f}'.format(accuracy_train_3[counter_lambda]) + '\n')
    f.write('\t\t-Valid Accuracy:    #Group0:' + '{:.3f}'.format(accuracy_valid_0[counter_lambda]) + '    #Group1:' + '{:.3f}'.format(accuracy_valid_1[counter_lambda]) + '    #Group2:' + '{:.3f}'.format(accuracy_valid_2[counter_lambda]) + '    #Group3:' + '{:.3f}'.format(accuracy_valid_3[counter_lambda]) + '\n\n')
f.close()


# 9. Prediction of test data

# Load test data
DATA_TEST_PATH = os.path.join(parentDir, 'data/test.csv')
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

# divide the test data corresponding to the feature "PRI_jet_num"
X_test_0, _ , ids_test_0 = divide_database([], tX_test,ids_test , PRI_jet_num = 0)
X_test_1, _ , ids_test_1 = divide_database([], tX_test,ids_test , PRI_jet_num = 1)
X_test_2, _ , ids_test_2 = divide_database([], tX_test,ids_test , PRI_jet_num = 2)
X_test_3, _ , ids_test_3 = divide_database([], tX_test,ids_test , PRI_jet_num = 3)

# change the shape of ids 
ids_test_0 = ids_test_0[:,np.newaxis]
ids_test_1 = ids_test_1[:,np.newaxis]
ids_test_2 = ids_test_2[:,np.newaxis]
ids_test_3 = ids_test_3[:,np.newaxis]

# cleaning test data 
X_test_0,_ = cleaning_data([], X_test_0, irrelevant_feature_columns=[4,5,6,12,22,23,24,25,26,27,28,29], replace_missing_data = True)
X_test_1,_ = cleaning_data([], X_test_1, irrelevant_feature_columns=[4,5,6,12,22,26,27,28], replace_missing_data = True)
X_test_2,_ = cleaning_data([], X_test_2, irrelevant_feature_columns=[22], replace_missing_data = True)
X_test_3,_ = cleaning_data([], X_test_3, irrelevant_feature_columns=[22], replace_missing_data = True)


# correlation
X_test_0,_ = cleaning_data([], X_test_0, [3], replace_missing_data = False)
X_test_1,_ = cleaning_data([], X_test_1, [18], replace_missing_data = False)


# test set is standardized (z-score normalization).
_, X_test_0 = standardize(X_tr_0, X_test_0)
_, X_test_1 = standardize(X_tr_1, X_test_1)
_, X_test_2 = standardize(X_tr_2, X_test_2)
_, X_test_3 = standardize(X_tr_3, X_test_3)

    
# adding the intercept term
X_test_0 =  np.c_[np.ones(X_test_0.shape[0]),X_test_0]
X_test_1 =  np.c_[np.ones(X_test_1.shape[0]),X_test_1]
X_test_2 =  np.c_[np.ones(X_test_2.shape[0]),X_test_2]
X_test_3 =  np.c_[np.ones(X_test_3.shape[0]),X_test_3]


y_test_pred_0 = predict_logistic(X_test_0, w_0[np.argmax(accuracy_valid_0)])
y_test_pred_1 = predict_logistic(X_test_1, w_1[np.argmax(accuracy_valid_1)])
y_test_pred_2 = predict_logistic(X_test_2, w_2[np.argmax(accuracy_valid_2)])
y_test_pred_3 = predict_logistic(X_test_3, w_3[np.argmax(accuracy_valid_3)])


ids_con = np.concatenate((ids_test_0, ids_test_1, ids_test_2, ids_test_3))
y_pred_con = np.concatenate((y_test_pred_0, y_test_pred_1, y_test_pred_2, y_test_pred_3))
y_pred_ids_con = np.concatenate((y_pred_con, ids_con),axis=1)
y_pred_ids_con = y_pred_ids_con[y_pred_ids_con[:, 1].argsort()]
y_pred_ids_con = y_pred_ids_con.astype('int')

OUTPUT_PATH = os.path.join(parentDir, 'data/submission_base_model_logistic.csv')
create_csv_submission(ids_test, y_pred_ids_con[:,0], OUTPUT_PATH)