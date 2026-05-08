# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 11:11:58 2026

@author: Mira
"""
from Final_GLA import GLA
from Bit_calculations import X_bits,C_16bit,indencies_bits
from Initializations_updated import Random_partition
from Vibration_data import Training_matrix
import numpy as np

X=Training_matrix
K=2
Titel='random Partition'
Epsilon=0.001
epsilon=0.001
split_epsilon=10
max_iterations=10000

def run_GLA(X,K,Epsilon,max_iterations,Titel):
    Initialization=Random_partition(X, K)
    C,D,_,_,d,iteration = GLA(X, Initialization, Epsilon, max_iterations)
    Bits_X=X_bits(X)
    Bits_after=C_16bit(C)+indencies_bits(D, K)+128+128
    Procent=((Bits_after-Bits_X)/Bits_X)*100
    """
    print(Titel)
    print('Number of codevectors:',K)
    print('Average distortion:',d)
    print('Iterations:',iteration)
    print('Compression:')
    print('-'*20)
    #print('Bits for training matrix X:',Bits_X)
    print('Total bits after compression:',Bits_after)
    print('Percentage change:',Procent)
    """
    return d, iteration, Bits_X, Bits_after, Procent

#run_GLA(X, K, Epsilon, max_iterations, Titel)


def mean_run_GLA(number_of_runs):
    distortions = []
    iterations = []
    bits_after_list = []
    procent_list = []

    for _ in range(number_of_runs):
        d, iteration, Bits_X, Bits_after, Procent = run_GLA(
            X, K, Epsilon, max_iterations, "Run")

        distortions.append(d)
        iterations.append(iteration)
        bits_after_list.append(Bits_after)
        procent_list.append(Procent)
    print(Titel)
    print('Number of codevectors:',K)
    print("Average over", number_of_runs, "runs")
    print("Average distortion:", np.mean(distortions), "±", np.std(distortions, ddof=1))
    print("Average iterations:", np.mean(iterations), "±", np.std(iterations, ddof=1))
    print("Average bits after:", np.mean(bits_after_list))
    print("Average percentage change:", np.mean(procent_list))
mean_run_GLA(100)

