# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 08:39:17 2026

@author: Mira
"""

import matplotlib.pyplot as plt

K=[2,3,4,6,8,12,16,24,32,48,64]
Ad=[208463293,181619098,160202404,134518655,118533827,106397292,98502237,89949650,84067498,78508410,76338387]

K_optimal = 12
Ad_optimal = 106397292

plt.plot(K, Ad, marker='o')
plt.xlabel("K")
plt.ylabel("Average distortion")

plt.scatter(K_optimal, Ad_optimal,color='red', s=120)
plt.axvline(x=K_optimal,color='red', linestyle='--')
plt.axhline(y=Ad_optimal,color='red', linestyle='--')

# Tekst annotation
plt.text(K_optimal+6, Ad_optimal+10000000, 'Knee point (K ≈ 12)')
plt.grid()
plt.suptitle("Average Distortion vs. Number of Code Vectors (K)", fontsize=14)
plt.title("Random partition", fontsize=10)
plt.show()