# Import libraries
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd   # NEW

# Load dataset from CSV file
data = pd.read_csv("lab11-K-MeanData.csv")

# Convert to numpy array (important for KMeans)
X = data.values

# Create K-Means model (K = 3)
kmeans = KMeans(n_clusters=3, random_state=42)

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

# Plot centroids
plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200)

# Labels and title
plt.xlabel("Age")
plt.ylabel("Spending Score")
plt.title("K-Means Clustering (CSV Dataset)")

plt.show()