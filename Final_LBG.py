# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 09:53:58 2026

@author: Mira

This script performs the Linde-Buzo-Gray (LBG) algorithm based on a
given training matrix X.

"""
import numpy as np
from Vibration_data import Training_matrix
import time

X=Training_matrix
K=16
split_epsilon=5
epsilon=0.001
max_iterations=10000

def GLA_LBG(X, C, epsilon, max_iterations):
    """
    Performs GLA on matrix X based on the inputs; Training matrix X, 
    initialization method, Epsilon and max-iterations. The function returns
    the codebook, average distortion and the number of iterations.
    """
    prev_d = np.inf
    N = X.shape[1]

    for it in range(max_iterations):

        D = np.sum((X[:, :, None] - C[:, None, :]) ** 2, axis=0)
        labels = np.argmin(D, axis=1)

        K = C.shape[1]

        A = np.zeros((K, N))
        A[labels, np.arange(N)] = 1

        AAT = A @ A.T
        AAT_inv = np.linalg.pinv(AAT + 1e-10 * np.eye(K))
        XAT = X @ A.T

        C = XAT @ AAT_inv

        X_hat = C @ A
        d = np.mean(np.sum((X - X_hat) ** 2, axis=0))

        if abs(prev_d - d) < epsilon:
            break

        prev_d = d
    
    return C, d, it + 1


def LBG(X, K, epsilon, split_epsilon, max_iterations):
    """
    Performs LBG on matrix X based on the inputs; Training matrix X, 
    initialization method, Epsilon, split-epsilon and max-iterations.
    LBG returns the final codebook, the average distortion and the
    number of LBG iterations.
    """
    C = np.mean(X, axis=1, keepdims=True)#Initialization
    LBG_iters = 0

    while C.shape[1] < K:
        C = np.hstack([
            C + split_epsilon, 
            C - split_epsilon 
        ])

        C, d, iters = GLA_LBG(X, C, epsilon, max_iterations)
        LBG_iters += 1
        print(iters)

    return C, d, LBG_iters

def run_LBG(X,K,epsilon, split_epsilon,max_iterations):
    """
    Runs LBG and prints the results.
    """
    C, d, LBG_iters=LBG(X, K, epsilon, split_epsilon, max_iterations)
    
    print('LBG')
    print('Number of codevectors:',K)
    print('Average distortion:',d)
    print('LBG_iterations:',LBG_iters)
    
    #return d, LBG_iters

run_LBG(X, K, split_epsilon, epsilon, max_iterations)

assignment_distance_counter = 0

def dist_assign(a, b):
    """
    The function calculates the euclidean norm and counts the 
    number of distance calculations.The input a and b is the vectors
    we want to calculate the distance between.
    """
    global assignment_distance_counter # Specifies that assignment_distance_counter is a global counter and not local.
    assignment_distance_counter += 1
    return np.linalg.norm(a - b)

def LBG_assignment(X, C):
    """
    This functions is calculating the gamma-encoders. The purpose of 
    the function is to investigate how many distance calculation LBG uses 
    in the assignment and the runtime.
    The input is the training matrix X and the final codebook C.
    The function returns the gamma-encoders for each training vector 
    (This is not used for anything). Only the runtime and number of distance calculations.
    """
    N = X.shape[1]
    K = C.shape[1]
    labels = np.zeros(N, dtype=int)
    for i in range(N):
        best_k = 0
        best_d = float("inf")
        for k in range(K):
            d = dist_assign(X[:, i], C[:, k]) 
            if d < best_d:
                best_d = d
                best_k = k
        labels[i] = best_k
    return labels

#Run LBG
C, _, _ = LBG(X, K, epsilon, split_epsilon, max_iterations)

#Reset counter
assignment_distance_counter = 0

#Start time and counter
t0 = time.perf_counter()

#Run the assignment
labels_lbg = LBG_assignment(X, C)

#Stop time and counter
t1 = time.perf_counter()

#Print results
print("LBG assignment time:", t1 - t0)
print("LBG assignment distances:", assignment_distance_counter)
