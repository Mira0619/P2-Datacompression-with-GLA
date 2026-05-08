# -*- coding: utf-8 -*-
"""
Created on Thu May  7 11:43:29 2026

@author: Mira
"""

import numpy as np
import os
from scipy.fft import fft, fftfreq
from scipy.io import loadmat

Faulty_path = r"C:\Users\Mira\Skrivebord\DATAP2\archive\Faulty"
Healthy_path = r"C:\Users\Mira\Skrivebord\DATAP2\archive\Healthy"
folder_path = [Healthy_path, Faulty_path]
selected_class = "Healthy" 

# Sampling parameters
fs = 1000
T = 1 / fs

all_rows = []
labels = []
filenames = []
channels = []
example_signal = None
example_label = None

for path in folder_path:
    label = os.path.basename(path)

    for file in os.listdir(path):
        if file.endswith('.mat'):
            full_path = os.path.join(path, file)
            data = loadmat(full_path)

            key = next(k for k in data.keys() if not k.startswith('__'))
            signal = data[key]

            if label == selected_class and example_signal is None:
                example_signal = signal
                example_label = label
                
            if signal.ndim == 2 and signal.shape[1] >= 3:
                N = signal.shape[0]
                xf = fftfreq(N, T)[:N // 2]

                for ch in range(3):
                    raw = signal[:, ch]
                    yf = fft(raw)
                    mag = 2.0 / N * np.abs(yf[:N // 2])

                    # Gem kun selve feature-vectoren
                    all_rows.append(mag)

                    # Gem metadata separat (valgfrit)
                    labels.append(label)
                    filenames.append(file)
                    channels.append(ch)

# Konverter til NumPy array
X = np.array(all_rows)

X_min = X.min()
X_max = X.max()

X_scaled = (X / np.max(X) * 32767).astype(np.int16)

Training_matrix=X_scaled

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

fs = 1000
t = np.arange(example_signal.shape[0]) / fs

plt.figure(figsize=(10, 5))
plt.plot(t, example_signal[:, 0])


plt.xlabel("Time (sec)")
plt.ylabel("Amplitude")
plt.grid()
plt.xlim(0, 5)
plt.ylim(-3, 3)
plt.tight_layout()
plt.show()


signal = example_signal[:, 0]
N = len(signal)

yf = fft(signal)
xf = fftfreq(N, 1/fs)[:N//2]
amplitude = 2.0 / N * np.abs(yf[:N//2])

plt.figure(figsize=(10, 5))
plt.plot(xf, amplitude)

plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.grid()
plt.xlim(0, 500)
plt.ylim(-0.05, 0.85)
plt.tight_layout()
plt.show()