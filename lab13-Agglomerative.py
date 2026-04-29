# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Load dataset from CSV
data = pd.read_csv("lab13.csv")
X = data.values

# Create Agglomerative model
model = AgglomerativeClustering(n_clusters=3, linkage='ward')

# Fit and predict
labels = model.fit_predict(X)

# Print clusters
print("Cluster Labels:", labels)

# Plot clusters
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.xlabel("X values")
plt.ylabel("Y values")
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