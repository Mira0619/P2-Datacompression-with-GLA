# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 09:53:58 2026

@author: Mira
"""
import numpy as np

def GLA_LBG(X, C, epsilon, max_iterations):

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

    C = np.mean(X, axis=1, keepdims=True)
    total_iters = 0

    while C.shape[1] < K:

        sigma = np.std(X, axis=1, keepdims=True)

        C = np.hstack([
            C + split_epsilon * sigma,
            C - split_epsilon * sigma
        ])

        C, d, iters = GLA_LBG(X, C, epsilon, max_iterations)
        total_iters += iters

    return C, d, total_iters

def run_LBG(X,K,split_epsilon,epsilon,max_iterations):
    C, d, total_iterations=LBG(X, K, epsilon, split_epsilon, max_iterations)
    
    print(Titel)
    print('Number of codevectors:',K)
    print('Average distortion:',d)
    print('Iterations:',total_iterations)
    
    return d, total_iterations
run_LBG(X, K, split_epsilon, epsilon, max_iterations)


def mean_run_LBG(number_of_runs):
    distortions = []
    iterations = []

    for _ in range(number_of_runs):
        d, iteration, = run_LBG(X, K, split_epsilon, epsilon, max_iterations)
        distortions.append(d)
        iterations.append(iteration)
    print(Titel)
    print('Number of codevectors:',K)
    print("Average over", number_of_runs, "runs")
    print("Average distortion:", np.mean(distortions))
    print("Average iterations:", np.mean(iterations))
    

