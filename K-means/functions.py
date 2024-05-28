import numpy as np

# Read the dataset
def load_data(filename):
    data = np.loadtxt(filename, delimiter=',')
    return data

# Read the initial centroids
def load_centroids(filename):
    centroids = np.loadtxt(filename, delimiter=',')
    return centroids

# Calculate the Euclidean distance
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))