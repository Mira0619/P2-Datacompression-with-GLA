# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 08:50:46 2026

@author: Mira
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
    global anchor_distance_counter
    anchor_distance_counter += 1
    return np.linalg.norm(a - b)


def dist_bf(a, b):
    global bf_distance_counter
    bf_distance_counter += 1
    return np.linalg.norm(a - b)

# Initialization + GLA
C0 =Random_partition(X, K)
C_final, _, A, _,_, _ = GLA(X, C0, 0.05, 1000)



# Build clusters
def build_sorted_clusters(X, C, A):
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



# Codevector distances
def build_cluster_centroid_distances(C):
    K = C.shape[1]
    dist_matrix = np.zeros((K, K))

    for i in range(K):
        for j in range(K):
            dist_matrix[i, j] = np.linalg.norm(C[:, i] - C[:, j])

    return dist_matrix



# Anchor prunning
def anchor_pruning_fast(X, C, clusters, M=1):

    n = X.shape[1]
    K = C.shape[1]

    centroid_dist = build_cluster_centroid_distances(C)

    new_assignments = np.zeros(n, dtype=int)

    for i in range(K):

        if len(clusters[i]) == 0:
            continue

        candidate_centroids = np.argsort(centroid_dist[i])[1:M+1]

        for (idx, _) in clusters[i]:

            x = X[:, idx]

            best = i
            best_dist = dist_anchor(x, C[:, i])

            for j in candidate_centroids:

                d_x_cj = dist_anchor(x, C[:, j])

                if d_x_cj < best_dist:
                    best = j
                    best_dist = d_x_cj

            new_assignments[idx] = best

    A_new = np.zeros((K, n), dtype=int)
    A_new[new_assignments, np.arange(n)] = 1

    return A_new



# Brute force
def brute_force_all(X, C):
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
print("\n--- TIMING ---")
print("Anchor build time:", t1 - t0)
print("Anchor query time:", t3 - t2)
print("Anchor total time:", t3 - t0)

print("Brute force time:", t5 - t4)

print("\n--- CORRECTNESS ---")
print("Anchor == BF:", np.mean(anchor_labels == bf_labels))

print("\n--- DISTANCE COMPARISON ---")
print("Brute force distances:", bf_distance_counter)
print("Anchor-based distances:", anchor_distance_counter)
print("Reduction (%):", 100 * (1 - anchor_distance_counter / bf_distance_counter))

# Distance counters
kd_distance_counter = 0
bf_distance_counter = 0


def dist_kd(a, b):
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
print("\n--- TIMING ---")
print("KD-tree build time:", t1 - t0)
print("KD-tree query time:", t3 - t2)
print("Brute force time:", t5 - t4)

print("\n--- CORRECTNESS ---")
print("KD == brute:", np.array_equal(kd_assignments, bf_assignments))

print("\n--- DISTANCE COMPARISON ---")
print("Brute force distances:", bf_distance_counter)
print("KD-tree distances:", kd_distance_counter)
print("Reduction (%):", 100 * (1 - kd_distance_counter / bf_distance_counter))
