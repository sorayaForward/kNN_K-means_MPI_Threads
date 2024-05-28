import numpy as np
import time
import threading
from functions import load_data, load_centroids, euclidean_distance

K = 7  # Number of clusters
MAX_ITER = 5  # Maximum number of iterations
FILENAME1 = "./datasets/covtype.csv"  # File containing the dataset
FILENAME2 = "./datasets/centroids.csv"  # File containing the k centroids

# Assign clusters to data points
def assign_clusters(data, centroids, labels, start, end):
    print("Start calculating distances")
    for i in range(start, end):
        distances = [euclidean_distance(data[i], centroid[:-1]) for centroid in centroids]  # Exclude label column
        labels[i] = np.argmin(distances)  # Assign the closest centroid

if __name__ == "__main__":
    data = load_data(FILENAME1)
    centroids = load_centroids(FILENAME2)
    print("Instances and centroids loaded")
    data_size = data.shape[0]
    instances_columns = data.shape[1]
    
    if instances_columns != centroids.shape[1] - 1:
        raise ValueError("Data columns and Centroid columns (excluding label) do not match")

    num_threads = 9  # Number of threads

    start_time = time.time()  # Start execution time

    threads = []
    local_labels = np.zeros(data_size, dtype=int)

    for iter_num in range(MAX_ITER):
        # Split data into chunks for each thread
        chunk_size = (data_size + num_threads - 1) // num_threads  # Ceiling division
        for i in range(num_threads):
            start_index = i * chunk_size
            end_index = min(start_index + chunk_size, data_size)
            thread = threading.Thread(target=assign_clusters, args=(data, centroids, local_labels, start_index, end_index))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()
        threads = []

        # Initialize local sums and counts
        local_sums = np.zeros((K, instances_columns))
        local_counts = np.zeros(K, dtype=int)

        # Accumulate sums and counts for each cluster
        for i in range(data_size):
            local_sums[local_labels[i]] += data[i]
            local_counts[local_labels[i]] += 1

        # Update centroids based on local sums and counts
        old_centroids = centroids[:, :-1].copy()  # Store the old centroids for comparison
        for i in range(K):
            if local_counts[i] > 0:
                centroids[i, :-1] = local_sums[i] / local_counts[i]  # Update centroids excluding labels

        # Check convergence: if centroids haven't changed, break the loop
        if np.allclose(old_centroids, centroids[:, :-1]):
            print(f"Convergence reached at iteration {iter_num}")
            break

    # Measure execution time
    execution_time = time.time() - start_time
    with open('execution_time_th.csv', 'a') as f:
        f.write(f"{num_threads},{execution_time}\n")

    # Print final centroids
    print("Final centroids:")
    print(centroids)

    # Print which instances belong to which centroid
    print("Instance assignments to centroids:")
    for i in range(data_size):
            print(f"Instance {i} is in cluster {local_labels[i]}")
    print(f"Convergence reached at iteration {iter_num}")