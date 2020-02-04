import numpy as np
from proj1_helpers import logistic_function


def compute_gradient_GD(y, tx, w):
    """
    Compute the full gradient of the Mean-Square-Error (MSE) loss function w.r.t the vector of parameters 'w'
    for linear models.
    
    INPUT:
    y  = target values as a vector of the shape (N,1) in which 'N' is the number of data points.
    tx = design matrix, where each row is a sample and the columns are the features.
         For example, for a database containing N samples, and D features, "tx" has the shape: (i) without including the
         offset term (N,D); and with including the offset term (N,D+1).
    w = parameters of the linear model as a vector of the shape 
        (D,1) or (D+1,1) (if a column of '1's is added to the design matrix), where D is the number of features.
        
    OUTPUT:
    Return the full gradient of MSE loss function.
    """
    e = y - tx @ w 
    gradient = - (tx.T @ e) / tx.shape[0]   # full gradient
    return gradient


def compute_gradient_SGD(y_n, tx_n, w):
    """
    Compute a stochastic gradient from just one data point 'n' and its corresponding 'y_n' label.
    
    INPUT:
    y_n  = target value, corresponding to for instance, the 'n'th entry of target vector.
    tx_n = the row 'n' of the design matrix of the shape (1,D).
    w = parameters of the linear model as a rank 1 numpy array of the shape 
        (D,1) or (D+1,1) (if a column of '1's is added to the design matrix), where D is the number of features.
        
    OUTPUT:
    Return the stochastic gradient of MSE loss function (the gradient of loss function corresponding
     to the 'n'th data point).
    """
    e = y_n - tx_n @ w
    gradient = - tx_n.T @ e
    return gradient


def compute_gradient_logistic_SGD(y_n, tx_n, w, lambda_=0):
    """
    Compute a stochastic gradient from just one data point 'n' and its corresponding 'y_n' label
    in logistic regression problem.
    
    INPUT:
    y_n     = target value, corresponding to for instance, the 'n'th entry of target vector.
    tx_n    = the row 'n' of the design matrix of the shape (1,D).
    w       = parameters of the linear model as as a vector of the shape 
             (D,1) or (D+1,1) (if a column of '1's is added to the design matrix), where D is the number of features.
    lambda_ = regularization strength parameter (default value = 0)
    OUTPUT:
    Return the stochastic gradient of logistic regression loss function 
    (the gradient of loss function corresponding to the 'n'th data point).
    """
    e = logistic_function(tx_n @ w) - y_n
    gradient = (tx_n.T * e) + lambda_ * w
    return gradient


def compute_gradient_logistic_GD(y, tx, w, lambda_=0):
    """
    Compute the full gradient of cross-entropy loss function in logistic
    regression problem.
    
    INPUT:
    y  = target values as a vector of the shape (N,1) in which 'N' is the number of data points.
    tx = design matrix, where each row is a sample and the columns are the features.
    w  = parameters of the linear model as as a vector of the shape 
        (D,1) or (D+1,1) (if a column of '1's is added to the design matrix), where D is the number of features.
    lambda_ = regularization strength parameter (default value = 0)
    OUTPUT:
    Return the full gradient of logistic regression loss function 
    """
    e = logistic_function(tx @ w) - y
    gradient = (tx.T @ e) + lambda_ * w
    return gradient
