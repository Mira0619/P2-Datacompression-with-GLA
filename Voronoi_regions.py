# -*- coding: utf-8 -*-
"""
Created on Mon May 11 09:46:02 2026

@author: matil
Plots codevectors, training vectors and voronoi regions for visual 
representation.
"""
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure

# Code vectors 
code_vectors = np.array([
    [2, 2],
    [8, 3],
    [5, 8],
    [3, 6],
    [7, 7]
])

# Data points
data_points = np.random.rand(120, 2) * 10

# Create a grid over the space
grid_x, grid_y = np.meshgrid(
    np.linspace(0, 10, 600),
    np.linspace(0, 10, 600)
)
grid_points = np.c_[grid_x.ravel(), grid_y.ravel()]

# Assign each grid point to nearest code vector
distances = np.linalg.norm(
    grid_points[:, None, :] - code_vectors[None, :, :],
    axis=2
)
regions = np.argmin(distances, axis=1).reshape(grid_x.shape)

# Plot
plt.figure(figsize=(7, 7))

# Draw boundaries between regions
for k in range(len(code_vectors)):
    mask = (regions == k).astype(float)
    contours = measure.find_contours(mask, 0.5)
    for contour in contours:
        plt.plot(contour[:, 1] * (10 / 600),
                 contour[:, 0] * (10 / 600),
                 color='black', linewidth=1)

# Plot data points
plt.scatter(data_points[:, 0], data_points[:, 1],
            color='gray', s=20, alpha=0.7)

# Plot code vectors
plt.scatter(code_vectors[:, 0], code_vectors[:, 1],
            color='blue', s=100, marker='o')

# Clean look
plt.xticks([])
plt.yticks([])
plt.grid(False)

plt.tight_layout()
plt.show()
