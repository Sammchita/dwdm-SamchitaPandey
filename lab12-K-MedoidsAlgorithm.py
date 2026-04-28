import numpy as np

# Sample dataset
X = np.array([
    [1, 2], [2, 3], [3, 4],
    [8, 7], [9, 8], [10, 9]
])

# Step 1: Choose K
k = 2

# Step 2: Initialize medoids randomly
medoid_indices = [0, 3]
medoids = X[medoid_indices]

def calculate_distance(a, b):
    return np.sum(np.abs(a - b))  # Manhattan distance

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
print("Clusters:")
for i, cluster in enumerate(clusters):
    print(f"Cluster {i+1}: {cluster}")