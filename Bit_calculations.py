# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 10:52:34 2026

@author: Mira
"""
import numpy as np

def X_bits(X):
    bits=16*X.shape[1]*X.shape[0]
    return bits
   

def C_16bit(C):
    # C forventes allerede at være int16
    C = C.astype(np.int16)
    # 16-bit repræsentation
    bits_C = C.size * 16
    return bits_C


def indencies_bits(D,K):
    Encoder_gamma = np.argmin(D, axis=0)
    bits_indices = Encoder_gamma.size * 16
    return bits_indices
    
