# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 10:26:21 2026

@author: Mira

Plot bitrate-distortion curve.
"""

import matplotlib.pyplot as plt

#Data that needs to be plottet
Ad=[208463293,181619098,160202404,134518655,118533827,106397292,98502237,89949650,84067498,78508410,76338387]
Bitrate=[0.0015152,0.0024014,0.0030303,0.0039168,0.0045455,0.0054318,0.0060606,0.0069472,0.0075758,0.0084622,0.0090909] 

#Plot
plt.plot(Bitrate, Ad, marker='o')
plt.xlabel("Bitrate")
plt.ylabel("Average distortion")

# Tekst 
plt.grid()
plt.suptitle("Rate-distortion curve", fontsize=14)
plt.title("Random partition", fontsize=10)
plt.show()