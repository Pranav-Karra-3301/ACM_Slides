

import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import time

# Seed for reproducibility
np.random.seed(42)

# Generate clusters
cluster1 = np.random.normal(loc=[2, 2], scale=0.5, size=(100, 2))
cluster2 = np.random.normal(loc=[8, 3], scale=0.5, size=(100, 2))
cluster3 = np.random.normal(loc=[5, 7], scale=0.5, size=(100, 2))


data = np.vstack([cluster1, cluster2, cluster3])

# Number of clusters
k = 5

# Initialize centers randomly
centers = data[np.random.choice(len(data), k, replace=False)]

# Function to assign clusters
def assign_clusters(data, centers):
    distances = np.sqrt(((data - centers[:, np.newaxis]) ** 2).sum(axis=2))
    return np.argmin(distances, axis=0)

# Function to update centers
def update_centers(data, labels, k):
    return np.array([data[labels == i].mean(axis=0) for i in range(k)])

# Function to run K-means and visualize updating
def kmeans_visualize(data, k, max_iters=100):
    # Initialize centers randomly
    centers = data[np.random.choice(len(data), k, replace=False)]

    # Create a new figure for each iteration
    plt.figure(figsize=(10, 6))

    for i in range(max_iters):
        # Assign clusters
        labels = assign_clusters(data, centers)

        # Clear previous plot
        plt.clf()

        # Plot data points with color based on cluster
        for j in range(k):
            cluster_points = data[labels == j]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {j + 1}')

        # Update centers
        new_centers = update_centers(data, labels, k)

        # Plot centers
        plt.scatter(centers[:, 0], centers[:, 1], color='black', marker='x', s=100, label='Old Centers')
        plt.scatter(new_centers[:, 0], new_centers[:, 1], color='red', marker='x', s=100, label='New Centers')

        # Check convergence
        if np.all(new_centers == centers):
            break

        centers = new_centers  # Update centers for next iteration

        # Plot customization
        plt.title(f'Iteration {i + 1}')
        plt.legend(loc='upper left')

        # Display the plot
        display.clear_output(wait=True)
        display.display(plt.gcf())
        time.sleep(0.5)

    plt.show()

# Run and visualize K-means
kmeans_visualize(data, k)

