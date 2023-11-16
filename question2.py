#Write a python program to implement k-means algorithms on a synthetic dataset.

import matplotlib.pyplot as plt   # Matplotlib for basic plotting
from sklearn.datasets import make_blobs  # make_blobs to generate synthetic data for clustering
from sklearn.cluster import KMeans

# Generate synthetic data with 300 samples, 2 features, 5 centers, and a standard deviation of 1.8
data = make_blobs(n_samples=300, n_features=2, centers=5, cluster_std=1.8, random_state=101)

# Create and fit a KMeans model with 5 clusters
kmeans = KMeans(n_clusters=5).fit(data[0])

# Display the cluster centers learned by KMeans
kmeans.cluster_centers_

# Create subplots to compare KMeans clustering results with the original data
fig ,(ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10, 6))

# Plot the data points colored by KMeans cluster assignments
ax1.set_title('K Means')
ax1.scatter(data[0][:, 0], data[0][:, 1], c=kmeans.labels_, cmap='brg')

# Plot the original data points colored by the true cluster assignments
ax2.set_title("Original")
ax2.scatter(data[0][:, 0], data[0][:, 1], c=data[1], cmap='brg')


#The code generates synthetic data with five clusters, visualizes it, and applies KMeans clustering with five clusters.
#It then displays the cluster centers and compares the clustering results with the original data through side-by-side scatter plots, illustrating how well KMeans captures the underlying patterns in the data.
