import numpy as np
import pandas as pd

# Load dataset from CSV
data = pd.read_csv("lab12.csv")
X = data.values

# Step 1: Choose K
k = 3

# Step 2: Initialize medoids (choose 3 different points)
medoid_indices = [0, 4, 8]
medoids = X[medoid_indices]

# Distance function (Manhattan Distance)
def calculate_distance(a, b):
    return np.sum(np.abs(a - b))

# Step 3–6: Iterate
for iteration in range(10):
    clusters = [[] for _ in range(k)]

    # Assign points to nearest medoid
    for point in X:
        distances = [calculate_distance(point, m) for m in medoids]
        cluster_index = np.argmin(distances)
        clusters[cluster_index].append(point)

    # Update medoids
    new_medoids = []
    for cluster in clusters:
        costs = []
        for candidate in cluster:
            cost = sum(calculate_distance(candidate, p) for p in cluster)
            costs.append(cost)
        new_medoids.append(cluster[np.argmin(costs)])

    new_medoids = np.array(new_medoids)

    # Stop if no change
    if np.array_equal(medoids, new_medoids):
        break

    medoids = new_medoids

# Output
print("Final Medoids:\n", medoids)
print("\nClusters:")
for i, cluster in enumerate(clusters):
    print(f"Cluster {i+1}: {cluster}")