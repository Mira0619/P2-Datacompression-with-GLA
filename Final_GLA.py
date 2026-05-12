# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 11:01:39 2026

@author: Mira

This script performs the generalized Lloyds algorithm(GLA) based on a
given training matrix X and corresponding initial codebook (Can be found 
with different initialization methods in Initializations_updated.py).
All functions in the script is based on the equations from 
subsection 3.3.1 (GLA expressed with linear algebra) in the project.
"""
import numpy as np

def Distance_matrix(X, C):
    """
    Calculates the distance matrix D(numpy.ndarray) based on the inputs;
    training set X(numpy.ndarray) and 
    codebook C(numpy.ndarray).
    """
    n = X.shape[1]

    Diag_XTX_T = np.sum(X * X, axis=0, keepdims=True)
    Diag_CTC = np.sum(C * C, axis=0, keepdims=True).T
    CTX = np.dot(C.T, X)

    D = (np.dot(np.ones((C.shape[1], 1)), Diag_XTX_T)
         - 2 * CTX
         + np.dot(Diag_CTC, np.ones((1, n))))

    return D

def Assignment_matrix(X, D, C):
    """
    Calculates the assignment matrix A (numpy.ndarray) based on 
    the inputs; training set X(numpy.ndarray), codebook C(numpy.ndarray)
    and distance matrix D(numpy.ndarray).
    """
    n = X.shape[1]
    Encoder_gamma = np.argmin(D, axis=0)

    A = np.zeros((C.shape[1], n), dtype=np.float64)
    A[Encoder_gamma, np.arange(n)] = 1

    return A


def Reconstruction_matrix(C, A):
    """
    Compute the reconstruktion matrix(numpy.ndarray) based on 
    codebook C (numpy.ndarray) and Assignment matrix A (numpy.ndarray). 
    Returns X_hat(numpy.ndarray).
    """
    return np.dot(C, A)


def Average_distortion(X, X_hat):
    """
    Calculates the average distortion(int). 
    The inputs are the training matrix X (numpy.ndarray) and 
    the reconstruction matrix X_hat (numpy.ndarray).
    """
    n = X.shape[1]
    return (1 / n) * np.sum((X - X_hat) ** 2)


def Update_codebook(X, A):
    """
    Updates the codebook C(numpy.ndarray) with the inputs; 
    training matrix X(numpy.ndarray) and assignment matrix A(numpy.ndarray).
    """
    AAT_inv = np.linalg.pinv(np.dot(A, A.T))
    XAT = np.dot(X, A.T)
    return np.dot(XAT, AAT_inv)

def to_int16(x):
    """
    Converts a matrix into a matrix with 16 bits per entry. In this 
    script the function is used scale the codebook C to 16 bits per entry. 
    """
    x = np.rint(x)                        
    x = np.clip(x, -32768, 32767)          
    return x.astype(np.int16)

def GLA(X, Initialization, Epsilon, max_iterations):
    """
    Performs GLA on matrix X based on the inputs; Training matrix X, 
    initialization method, Epsilon and max-iterations.The function returns 
    the final codebook(16 bits), the final distance matrix D,
    the final assignment matrix A, The reconstruction matrix X_hat
    , the final average distortion and the number of GLA iterations.
    """
    X = X.astype(np.float64)
    C = Initialization.astype(np.float64)

    prev_d = np.inf

    for iteration in range(max_iterations):

        D = Distance_matrix(X, C)
        A = Assignment_matrix(X, D, C)
        X_hat = Reconstruction_matrix(C, A)

        d = Average_distortion(X, X_hat)

        if abs(prev_d - d) < Epsilon:
            break

        C = Update_codebook(X, A)
        prev_d = d

    X_hat = Reconstruction_matrix(C, A)

    C_int16 = to_int16(C)
    X_hat_int16 = to_int16(X_hat)

    return C_int16, D, A, X_hat_int16, d, iteration + 1

