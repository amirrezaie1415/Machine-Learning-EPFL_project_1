# -*- coding: utf-8 -*-

import numpy as np
from proj1_helpers import *
from loss import *
from gradient import *


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using gradient descent (Using Mean-Square-Error (MSE) loss function).
    
    INPUT:
    y  =            target values as a vector of the shape (N,1) in which 'N' is the number of data points.
    tx =            design matrix, where each row is a sample and the columns are the features.
                    For example, for a database containing N samples, and D features, "tx" has the shape: (i) without
                    including the offset term (N,D); and with including the offset term (N,D+1).
    initial_w =     parameters of the linear model as a vector of the shape (D,1) or (D+1,1) (if a column of '1's is
                    added to the design matrix), where D is the number of features.
    max_iters =     number of steps to run.
    gamma     =     step-size
    
    OUTPUT:
    Return the last weight vector of the method, and the corresponding loss value (cost function).
    """
    w = initial_w
    for n_iter in range(max_iters):
        # compute gradient
        gradient = compute_gradient_GD(y, tx, w)
        # updating the weight vector
        w = w - gamma * gradient
        loss = compute_loss_mse(y, tx, w)  # compute loss
        print("least_squares_GD({bi}/{ti}): loss={l:.3f}, w0={w0:.3f}, w1={w1:.3f}".format(
            bi=n_iter+1, ti=max_iters, l=loss, w0=w[0, 0], w1=w[1, 0]))
    return w, loss


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using stochastic gradient descent (Using Mean-Square-Error (MSE) loss function).
    
    INPUT:
    y  =            target values as a vector of the shape (N,1) in which 'N' is the number of data points.
    tx =            design matrix, where each row is a sample and the columns are the features.
                    For example, for a database containing N samples, and D features, "tx" has the shape: (i) without
                    including the offset term (N,D); and with including the offset term (N,D+1).
    initial_w =     parameters of the linear model as a vector of the shape (D,1) or (D+1,1) (if a column of '1's is
                    added to the design matrix), where D is the number of features.
    max_iters =     number of epochs
    gamma     =     step-size
    
    OUTPUT:
    Return the last weight vector of the method, and the corresponding loss value (cost function).
    """
    epochs = max_iters
    w = initial_w
    for epoch in range(epochs):
        # step = number of weights update
        step = 1
        for y_n, tx_n in next_batch(y, tx):
            # compute gradient
            gradient = compute_gradient_SGD(y_n, tx_n, w)
            # update weights
            w = w - gamma * gradient
            # compute loss
            loss = compute_loss_mse(y, tx, w)
            print("least_squares_SGD"
                  "(epoch:{bi}/{ti}, step:{n_step}/{num_sample}): loss={l:.3f}, w0={w0:.3f}, w1={w1:.3f}".format(
                bi=epoch+1, ti=epochs,n_step=step, num_sample=y.shape[0], l=loss, w0=w[0, 0], w1=w[1, 0]))
            step += 1
    return w, loss


def least_squares(y, tx):
    """
    Least squares regression using normal equations
    
    INPUT:
    y  =            target values as a vector of the shape (N,1) in which 'N' is the number of data points.
    tx =            design matrix, where each row is a sample and the columns are the features.
                    For example, for a database containing N samples, and D features, "tx" has the shape: (i) without
                    including the offset term (N,D); and with including the offset term (N,D+1).
    
    OUTPUT:
    Return the last weight vector of the method, and the corresponding loss value (cost function).
    """
    # Solving the linear system of equations for Aw = B, where A = tx.T @ tx and B = tx.T @ y
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    loss = compute_loss_mse(y, tx, w)  # compute loss
    return w, loss


def ridge_regression(y, tx, lambda_):
    """
    Ridge regression using normal equations
    
    INPUT:
    y  =            target values as a vector of the shape (N,1) in which 'N' is the number of data points.
    tx =            design matrix, where each row is a sample and the columns are the features.
                    For example, for a database containing N samples, and D features, "tx" has the shape: (i) without
                    including the offset term (N,D); and with including the offset term (N,D+1).
    lambda_ =       regularization strength parameter
    
    OUTPUT:
    Return the last weight vector of the method, and the corresponding loss value (cost function).
    """
    # Solving the linear system of equations for Aw = B, where A = (tx.T @ tx + 2N lambda I) and B = tx.T @ y
    w = np.linalg.solve(tx.T @ tx + 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1]), tx.T @ y)
    loss = compute_loss_mse(y, tx, w)  # compute loss
    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Logistic regression using SGD.
    
    INPUT:
    y  =            target values as a vector of the shape (N,1) in which 'N' is the number of data points.
    tx =            design matrix, where each row is a sample and the columns are the features.
                    For example, for a database containing N samples, and D features, "tx" has the shape: (i) without
                    including the offset term (N,D); and with including the offset term (N,D+1).
    initial_w =     parameters of the linear model as a vector of the shape (D,1) or (D+1,1)
                    (if a column of '1's is added to the design matrix), where D is the number of features.
    max_iters =     number of epochs
    gamma     =     step-size
    
    OUTPUT:
    Return the last weight vector of the method, and the corresponding loss value (cost function).
    """
    epochs = max_iters
    w = initial_w
    for epoch in range(epochs):
        # step = weight update step number
        step = 1
        for y_n, tx_n in next_batch(y, tx):
            # compute gradient
            gradient = compute_gradient_logistic_SGD(y_n, tx_n, w)
            # update weights
            w = w - gamma * gradient
            # compute loss
            loss = compute_loss_logistic(y, tx, w)
            # print the status every 1000 steps
            if step % 1000 == 0 or step == tx.shape[0]:
                print("logistic_regression_SGD(epoch:{bi}/{ti}), step:{n_step}/{num_sample}): loss={l:.8f}".format(
                  bi=epoch+1, ti=epochs, n_step=step, num_sample=y.shape[0], l=loss))
            step += 1
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Regularized logistic regression using using GD.
    
    INPUT:
    y  =            target values as a vector of the shape (N,1) in which 'N' is the number of data points.
    tx =            design matrix, where each row is a sample and the columns are the features.
                    For example, for a database containing N samples, and D features, "tx" has the shape: (i) without
                    including the offset term (N,D); and with including the offset term (N,D+1).
    lambda_ =       regularization strength parameter
    initial_w =     parameters of the linear model as a vector of the shape  (D,1) or (D+1,1) (if a column of '1's is
                    added to the design matrix), where D is the number of features.
    max_iters =     number of epochs (one epoch is when the entire data set is passed)
    gamma     =     step-size
    
    OUTPUT:
    Return the last weight vector of the method, and the corresponding loss value (cost function).
    """
    w = initial_w
    # print loss value for the initial weight
    loss_0 = compute_loss_logistic(y, tx, w)
    print("(epoch: {bi}/{ti}): training loss={l:.1f}".format(bi=0, ti=max_iters, l=loss_0))
    for n_iter in range(max_iters):
        # compute gradient
        gradient = compute_gradient_logistic_GD(y, tx, w, lambda_)
        # updating the weight vector
        w = w - gamma * gradient
        loss = compute_loss_logistic(y, tx, w)
        # if total number of steps is larger than 1000, print the status every 500 steps
        if max_iters > 1000:
            if n_iter+1 in range(1,max_iters, 500):
                print("(epoch: {bi}/{ti}): training loss={l:.1f}".format(bi=n_iter+1,ti=max_iters, l=loss))
            if n_iter+1 == max_iters:
                print("(epoch: {bi}/{ti}): training loss={l:.1f}".format(bi=n_iter+1,ti=max_iters, l=loss))
        else:  # otherwise print the status at every step
            print("(epoch: {bi}/{ti}): training loss={l:.1f}".format(bi=n_iter+1,ti=max_iters, l=loss))
    return w,loss

