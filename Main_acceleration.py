# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 08:50:46 2026

@author: Mira

This script is used to execute bruteforce, kd-trees and anchor-based pruning, 
based on the training matrix X and the finale codebook C. The script uses
random seed 42, so that the two methods can be compared. 

"""

import numpy as np
import time

np.random.seed(42)

from Vibration_data import Training_matrix
from Initializations_updated import Random_partition
from Final_GLA import GLA


X = Training_matrix
K = 16


# Distance counters
anchor_distance_counter = 0
bf_distance_counter = 0


def dist_anchor(a, b):
    """
    The function calculates the euclidean norm a  nd counts the 
    number of distance calculations in the anchor-based pruning.
    The input a and b is the vectors we want to calculate the distance between.
    """
    global anchor_distance_counter
    anchor_distance_counter += 1
    return np.linalg.norm(a - b)


def dist_bf(a, b):
    """
    The function calculates the euclidean norm a  nd counts the 
    number of distance calculations with Brute force.
    The input a and b is the vectors we want to calculate the distance between.
    """
    global bf_distance_counter
    bf_distance_counter += 1
    return np.linalg.norm(a - b)

# Initialization + GLA
C0 =Random_partition(X, K)
C_final, _, A, _,_, _ = GLA(X, C0, 0.05, 1000)



# Build clusters
def build_sorted_clusters(X, C, A):
    """
    The function assigns the training vectors to the 
    codevectors and the training vectors in each cluster are
    sorted in increasing order according to the distance the corresponding codevector.
    """
    K = C.shape[1]
    clusters = {}

    for i in range(K):
        idx = np.where(A[i] == 1)[0]

        if len(idx) == 0:
            clusters[i] = []
            continue

        points = X[:, idx]
        dists = np.linalg.norm(points - C[:, i].reshape(-1, 1), axis=0)

        sorted_idx = np.argsort(dists)
        clusters[i] = list(zip(idx[sorted_idx], dists[sorted_idx]))

    return clusters


def build_cluster_centroid_distances(C):
    """
    The function computes the pairwise Euclidean distances between all cluster centroids
    and returns a distance matrix of size K × K, where K is the number of clusters.

    Each entry in the matrix represents the distance between 2 centroids.
    """
    K = C.shape[1]
    dist_matrix = np.zeros((K, K))

    for i in range(K):
        for j in range(K):
            dist_matrix[i, j] = np.linalg.norm(C[:, i] - C[:, j])

    return dist_matrix

#Anchor-based pruning
def anchor_pruning_fast(X, C, clusters, M=1):
    """
    The function performs anchor-based pruning to refine cluster assignments by comparing each
    data point only with a limited set of candidate codevectors instead of all codevectors.
    """
    n = X.shape[1]
    K = C.shape[1]
    
    centroid_dist = build_cluster_centroid_distances(C)# Compute pairwise distances between all centroids
    new_assignments = np.zeros(n, dtype=int) # Array to store new cluster assignments for each training vector
    # Loop over each codevector
    for i in range(K):
        if len(clusters[i]) == 0:
            continue
        candidate_centroids = np.argsort(centroid_dist[i])[1:M+1] # Select M nearest centroids to centroid i
        # Loop over all training vectors assigned to codevector i
        for (idx, _) in clusters[i]:
            x = X[:, idx]
            dist_i = dist_anchor(x, C[:, i])
            best = i
            best_dist = dist_i
            for j in candidate_centroids:
                if dist_i < 0.5 * centroid_dist[i, j]:#If this inequality is true then skip calculating the next distance.
                    continue
                d_x_cj = dist_anchor(x, C[:, j])
                if d_x_cj < best_dist:
                    best = j
                    best_dist = d_x_cj
            new_assignments[idx] = best
    A_new = np.zeros((K, n), dtype=int)
    A_new[new_assignments, np.arange(n)] = t1
    return A_new


def brute_force_all(X, C):
    """
    Computes assignment with brute force (Calculates all distances)
    """
    n = X.shape[1]
    K = C.shape[1]
    labels = np.zeros(n, dtype=int)
    
    for i in range(n):
        best = 0
        best_dist = float("inf")
        for j in range(K):
            d = dist_bf(X[:, i], C[:, j])
            if d < best_dist:
                best = j
                best_dist = d
        labels[i] = best
    return labels


def A_to_labels(A):
    """
    Converts an encoded assignment matrix into a label vector
    """
    return np.argmax(A, axis=0)


# Run

t0 = time.perf_counter()

clusters = build_sorted_clusters(X, C_final, A)
centroid_dist = build_cluster_centroid_distances(C_final)

t1 = time.perf_counter()

# Reset counter
anchor_distance_counter = 0

t2 = time.perf_counter()

A_anchor = anchor_pruning_fast(X, C_final, clusters)

t3 = time.perf_counter()

anchor_labels = A_to_labels(A_anchor)

# Brute force reset counter
bf_distance_counter = 0

t4 = time.perf_counter()

bf_labels = brute_force_all(X, C_final)

t5 = time.perf_counter()

#Results
print("TIMING")
print("Anchor build time:", t1 - t0)
print("Anchor query time:", t3 - t2)
print("Anchor total time:", t3 - t0)

print("Brute force time:", t5 - t4)

print("CORRECTNESS")
print("Anchor == BF:", np.mean(anchor_labels == bf_labels))

print("DISTANCE COMPARISON")
print("Brute force distances:", bf_distance_counter)
print("Anchor-based distances:", anchor_distance_counter)
print("Reduction (%):", 100 * (1 - anchor_distance_counter / bf_distance_counter))

# Distance counters
kd_distance_counter = 0
bf_distance_counter = 0


def dist_kd(a, b):
    """
    The function calculates the euclidean norm a  nd counts the 
    number of distance calculations in the KD-tree.
    The input a and b is the vectors we want to calculate the distance between.
    """
    global kd_distance_counter
    kd_distance_counter += 1
    return np.linalg.norm(np.array(a) - np.array(b))


# KD-tree structure
class KDNode:
    def __init__(self, point, index, left=None, right=None):
        self.point = point
        self.index = index
        self.left = left
        self.right = right


def build_kdtree(points, depth=0):
    """
    Builds a k-d tree from a list of points.The function constructs a k-d tree 
    by recursively selecting the median point along a dimension. At each depth, a
    splitting dimension is chosen, and the dataset is partitioned into
    left and right subtrees. The input is a list of tuples, where each element 
    is on the form (point,index). 
     - Point is a k-dimensional numpy array
     - Index is the original index of the point in the dataset
     
    The function returns a root node of the constructed k-d tree.
    """
    if not points:
        return None

    k = len(points[0][0])
    axis = depth % k

    points.sort(key=lambda p: p[0][axis])
    median = len(points) // 2

    return KDNode(
        point=points[median][0],
        index=points[median][1],
        left=build_kdtree(points[:median], depth + 1),
        right=build_kdtree(points[median + 1:], depth + 1)
    )


# KD search
def nearest_neighbor_search(node, target, depth=0, best=None):
    """
    Performs nearest neighbor search in a k-d tree. The function returns the 
    index of nearest codevectors and its distance to the target.
    """
    if node is None:
        return best

    k = len(target)
    axis = depth % k

    if target[axis] < node.point[axis]:
        next_branch = node.left
        opposite_branch = node.right
    else:
        next_branch = node.right
        opposite_branch = node.left

    best = nearest_neighbor_search(next_branch, target, depth + 1, best)

    dist = dist_kd(target, node.point)

    if best is None or dist < best[1]:
        best = (node.index, dist)

    hyperplane_dist = abs(target[axis] - node.point[axis])

    if best is None or hyperplane_dist < best[1]:
        best = nearest_neighbor_search(opposite_branch, target, depth + 1, best)

    return best

# KD tree assignment
def kd_tree_all(X, C):
    """
    Assigns each data point in X to its nearest centroid in C
    using a k-d tree for efficient nearest codevector search.
    """
    points = [(C[:, i], i) for i in range(C.shape[1])]
    root = build_kdtree(points)

    n = X.shape[1]
    assignments = np.zeros(n, dtype=int)

    for i in range(n):
        x = X[:, i]
        idx, _ = nearest_neighbor_search(root, x)
        assignments[i] = idx

    return assignments

# Brute force
def brute_force_all(X, C):
    """
    Computes assignment with brute force (Calculates all distances)
    """
    n = X.shape[1]
    K = C.shape[1]

    assignments = np.zeros(n, dtype=int)

    for i in range(n):
        best = 0
        best_dist = float("inf")

        for j in range(K):
            d = dist_bf(X[:, i], C[:, j])
            if d < best_dist:
                best = j
                best_dist = d

        assignments[i] = best

    return assignments

# Build KD tree
points = [(C_final[:, i], i) for i in range(C_final.shape[1])]

t0 = time.perf_counter()
root = build_kdtree(points)
t1 = time.perf_counter()

# KD quary
kd_distance_counter = 0

t2 = time.perf_counter()
kd_assignments = kd_tree_all(X, C_final)
t3 = time.perf_counter()

# Brute force
bf_distance_counter = 0

t4 = time.perf_counter()
bf_assignments = brute_force_all(X, C_final)
t5 = time.perf_counter()

# Results
print("Timing")
print("KD-tree build time:", t1 - t0)
print("KD-tree query time:", t3 - t2)

print("Correctness")
print("KD == brute:", np.array_equal(kd_assignments, bf_assignments))

print("Distance Comparison")
print("KD-tree distances:", kd_distance_counter)
print("Reduction (%):", 100 * (1 - kd_distance_counter / bf_distance_counter))
