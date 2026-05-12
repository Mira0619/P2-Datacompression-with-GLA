# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 10:52:34 2026

@author: Mira

This script contains three functions used to calculate 
bits in the training matrix, final codebook and the bits for the gamma indices.
"""
import numpy as np

def X_bits(X):
    """
    Calculates total number of bits in the training matrix.

    Parameters
    ----------
    X : numpy.ndarray
        Input training matrix
    Returns
    -------
    int: Total number of bits used to store all elements in X.
    """
    bits=16*X.shape[1]*X.shape[0]
    return bits


def C_16bit(C):
    """
    Calculates total number of bits in the final codebook.

    Parameters
    ----------
    C : numpy.ndarray
        Input final codebook matrix
    Returns
    -------
    int: Total number of bits used to store all elements in C.
    """
    C = C.astype(np.int16)#It is expected that all entries in C is 16 bits.
    bits_C = C.size * 16
    return bits_C


def indencies_bits(D):
    """
    Calculates how many bits it takes to store the gamma indices.

    Parameters
    ----------
    D : numpy.ndarray
        Input Distance matrix calculated in the function Distance_matrix(X,C)
        in the file Final_GLA.py
        
    Returns
    -------
    int: Total number of bits used to store all the gamma indices.
    """
    Encoder_gamma = np.argmin(D, axis=0)
    bits_indices = Encoder_gamma.size * 16
    return bits_indices
    
