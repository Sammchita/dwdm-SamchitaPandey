# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# Load dataset from CSV
data = pd.read_csv("lab14.csv")
X = data.values

# Create DBSCAN model
model = DBSCAN(eps=3, min_samples=2)

# Fit and predict
labels = model.fit_predict(X)

# Print cluster labels
print("Cluster Labels:", labels)

# Plot clusters
plt.scatter(X[:, 0], X[:, 1], c=labels)

plt.xlabel("X values")
plt.ylabel("Y values")
plt.title("DBSCAN Clustering")

plt.show()