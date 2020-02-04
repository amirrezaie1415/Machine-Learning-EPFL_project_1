import numpy as np
from proj1_helpers import logistic_function


def compute_loss_mse(y, tx, w):
    """
    Calculate the Mean-Squared-Error loss for linear models.
    INPUT:
    y  = target values as a vector of the shape (N,1) in which 'N' is the number of data points.
    tx = design matrix, where each row is a sample and the columns are the features.
         For example, for a database containing N samples, and D features, "tx" has the shape: (i) without including the
         offset term (N,D); and with including the offset term (N,D+1).
    w = parameters of the linear model as a vector of the shape 
        (D,1) or (D+1,1) (if a column of '1's is added to the design matrix), where D is the number of features.
    
    OUTPUT:
    Return the total loss  
    """
    e = y - tx @ w 
    loss = (e.T @ e) / (2 * tx.shape[0])
    return loss[0][0]


def compute_loss_logistic(y, tx, w):
    """
    Calculate the logistic regression loss.
    INPUT:
    y  = target values as as a vector of the shape (N,1) in which 'N' is the number of data points.
    tx = design matrix, where each row is a sample and the columns are the features.
         For example, for a database containing N samples, and D features, "tx" has the shape: (i) without including the
         offset term (N,D); and with including the offset term (N,D+1).
    w = parameters of the linear model as a vector of the shape 
        (D,1) or (D+1,1) (if a column of '1's is added to the design matrix), where D is the number of features.
    
    OUTPUT:
    Return the loss  
    """
    loss = np.sum(np.log(1 + np.exp(tx @ w))) - np.sum(y * tx @ w)
    return loss
