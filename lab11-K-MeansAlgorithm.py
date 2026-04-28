# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Sample dataset
X = np.array([
    [1, 2], [1.5, 1.8], [5, 8],
    [8, 8], [1, 0.6], [9, 11]
])

# Create K-Means model (K = 2)
kmeans = KMeans(n_clusters=2, random_state=42)

# Fit model
kmeans.fit(X)

# Get cluster labels
labels = kmeans.labels_

# Get centroids
centroids = kmeans.cluster_centers_

# Print results
print("Cluster Labels:", labels)
print("Centroids:\n", centroids)

# Plot clusters
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='X')
plt.title("K-Means Clustering")
plt.show()