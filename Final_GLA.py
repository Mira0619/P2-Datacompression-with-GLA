# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 11:01:39 2026

@author: Mira
"""

import numpy as np


def Distance_matrix(X, C):
    n = X.shape[1]

    Diag_XTX_T = np.sum(X * X, axis=0, keepdims=True)
    Diag_CTC = np.sum(C * C, axis=0, keepdims=True).T
    CTX = np.dot(C.T, X)

    D = (np.dot(np.ones((C.shape[1], 1)), Diag_XTX_T)
         - 2 * CTX
         + np.dot(Diag_CTC, np.ones((1, n))))

    return D

def Assignment_matrix(X, D, C):
    n = X.shape[1]
    Encoder_gamma = np.argmin(D, axis=0)

    A = np.zeros((C.shape[1], n), dtype=np.float64)
    A[Encoder_gamma, np.arange(n)] = 1

    return A


def Reconstruction_matrix(C, A):
    return np.dot(C, A)


def Average_distortion(X, X_hat):
    n = X.shape[1]
    return (1 / n) * np.sum((X - X_hat) ** 2)


def Update_codebook(X, A):
    AAT_inv = np.linalg.pinv(np.dot(A, A.T))
    XAT = np.dot(X, A.T)
    return np.dot(XAT, AAT_inv)

def to_int16(x):
    x = np.rint(x)                          # round
    x = np.clip(x, -32768, 32767)          # int16 range
    return x.astype(np.int16)

def GLA(X, Initialization, Epsilon, max_iterations):

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

