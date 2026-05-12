# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 08:38:10 2026

@author: Mira

The purpose of this script is to find the initial codebook. The 
functions in the script is based on Chapter 4 (Initialization) in the 
project.
"""
import numpy as np

def Forgy_initialization(X, K):
    """
    Initializes K codevectors using the Forgy algorithm.
    
    Chooses K random codevectors from the training matrix X. Returns the 
    initial codebook C_0.
    """
    n = X.shape[1]
    Random_columns = np.random.choice(n, size=K, replace=False)
    return X[:, Random_columns]

def Random_partition(X,K):
    """
    Initializes K codevectors using the Random partition algorithm.
    
    Splits training matrix X in K random clusters and calculates the mean 
    of each cluster and collects the mean of each cluster in the 
    initial codebook C_0.
    """
    n=X.shape[1]
    k=X.shape[0]
    Assign_cluster = np.random.randint(0, K, size=n)
    C_0_random=np.zeros((k, K))
    for cluster in range(K):
        columns_in_cluster = np.where(Assign_cluster == cluster)[0]
        C_0_random[:, cluster] = np.mean(X[:, columns_in_cluster], axis=1)
    return C_0_random

def NSA(X,K):
    """
    Initializes K codevectors using the Naive sharding algorithm(NSA).
    
    The functions sorts the columns in the training matrix X by the 
    sum of all entries in each column. Then the training matrix X is 
    split in K clusters and calculates the mean 
    of each cluster and collects the mean of each cluster in the 
    initial codebook C_0.
    """
    Sum_row = np.sum(X, axis=0, keepdims=True)
    X_sum = np.vstack((X, Sum_row))  
    X_sorted=X_sum[:,np.argsort(X_sum[-1])]
    X_shards = np.array_split(np.delete(X_sorted, -1, axis=0), K, axis=1)
    Initial_codevectors=[np.mean(shard,axis=1,keepdims=True) for shard in X_shards]
    Initial_codebook=np.hstack(Initial_codevectors)
    return Initial_codebook

def Kmeans_plus_plus(X, K):
    """
    Initializes K codevectors using the K-means++ algorithm.
    
    The function chooses the first codevector randomly from the training matrix X.
    Each subsequent codevector is selected from X with probability
    proportional to its squared distance from the nearest already
    chosen codevector. The codevectors is collected in the initial
    codebook C_0.
    """
    from Final_GLA import Distance_matrix
    n = X.shape[1]
    dim = X.shape[0]
    Codebook = X[:, np.random.choice(n)].reshape(dim, 1)
    for _ in range(1, K):
        D_matrix = Distance_matrix(X, Codebook)
        dist = np.min(D_matrix, axis=0)  
        dist = np.maximum(dist, 1e-12)   
        probs = dist / np.sum(dist)
        idx = np.random.choice(n, p=probs)
        Codebook = np.hstack((Codebook, X[:, idx].reshape(dim, 1)))
    return Codebook

def PCA_GLA_Initialization(X,K):
    """
    Initializes K codevectors using the PCA initialization algorithm.
    
    Performs PCA without dimension reduction on the training matrix X 
    and sort the training matrix X by the first row in increasing order. 
    Then the training matrix X is split in K clusters and calculates the mean 
    of each cluster and collects the mean of each cluster in the 
    initial codebook C_0.  
    
    """
    X_overline = np.mean(X, axis=1, keepdims=True) #The sample mean is calculated
    B = X - X_overline # Calculate centrered data matrix
    Cov = (1 / (X.shape[1] - 1)) * np.dot(B, B.T) # Covarians matrix
    Eigenvalues, Eigenvectors = np.linalg.eigh(Cov) #Calculate eigenvalues and eigenvectors
    idx = np.argsort(Eigenvalues)[::-1] # Sort eigenvalues descending
    P = Eigenvectors[:, idx] 
    y = np.dot(P.T, B) #Project data onto PCA basis
    y_sorted = y[:, np.argsort(y[0])]
    y_split = np.array_split(y_sorted, K, axis=1)
    C_pca = np.hstack([np.mean(shard, axis=1, keepdims=True) for shard in y_split])
    Codebook = np.dot(P, C_pca) + X_overline
    return Codebook

