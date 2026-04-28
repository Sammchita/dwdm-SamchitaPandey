# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Sample dataset
X = np.array([
    [1, 2], [2, 3], [3, 4],
    [8, 7], [9, 8], [10, 9]
])

# Create Agglomerative model
model = AgglomerativeClustering(n_clusters=2, linkage='ward')

# Fit and predict
labels = model.fit_predict(X)

# Print clusters
print("Cluster Labels:", labels)

# Plot clusters
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.title("Agglomerative Clustering")
plt.show()

# Create dendrogram
Z = linkage(X, method='ward')

plt.figure()
dendrogram(Z)
plt.title("Dendrogram")
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.show()