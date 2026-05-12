# -*- coding: utf-8 -*-
"""
Created on Tue May  5 13:41:41 2026

@author: Mira

The script is used to execute FFT on the 220 signal, and collects them 
in an array. The array is scaled so each entry is 16 bits.
"""
import numpy as np
import os
from scipy.fft import fft, fftfreq
from scipy.io import loadmat

#Load the signals
Faulty_path = r"C:\Users\Mira\Skrivebord\DATAP2\archive\Faulty"
Healthy_path = r"C:\Users\Mira\Skrivebord\DATAP2\archive\Healthy"
folder_path = [Healthy_path, Faulty_path]

# Sampling parameters
fs = 1000
T = 1 / fs

all_rows = []
labels = []
filenames = []
channels = []

#FFT - code from the article "An Expert System for Rotating Machine Fault Detection Using Vibration Signal Analysis", Link and reference in the project.
for path in folder_path:
    label = os.path.basename(path)

    for file in os.listdir(path):
        if file.endswith('.mat'):
            full_path = os.path.join(path, file)
            data = loadmat(full_path)

            key = next(k for k in data.keys() if not k.startswith('__'))
            signal = data[key]

            if len(all_rows) == 0:
                example_signal = signal
                example_label = label
                
            if signal.ndim == 2 and signal.shape[1] >= 3:
                N = signal.shape[0]
                xf = fftfreq(N, T)[:N // 2]

                for ch in range(3):
                    raw = signal[:, ch]
                    yf = fft(raw)
                    mag = 2.0 / N * np.abs(yf[:N // 2])
                    
                    all_rows.append(mag)

                    labels.append(label)
                    filenames.append(file)
                    channels.append(ch)

# convert to NumPy array
X = np.array(all_rows)

#Scale X to 16 bit per entry
X_min = X.min()
X_max = X.max()
X_scaled = (X / np.max(X) * 32767).astype(np.int16)

#The training matrix used in the project
Training_matrix=X_scaled

