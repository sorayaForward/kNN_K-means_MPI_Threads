from mpi4py import MPI
import numpy as np
import time
from functions import load_data, load_centroids, euclidean_distance

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


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

if rank == 0:
    print("Start loading")
    data = load_data(FILENAME1)
    centroids = load_centroids(FILENAME2)
    print("Instances and centroids loaded")
    data_size = data.shape[0]
    instances_columns = data.shape[1]
    
    if instances_columns != centroids.shape[1] - 1:
        raise ValueError("Data columns and Centroid columns (excluding label) do not match")
else:
    data = None
    centroids = None
    data_size = None
    instances_columns = None

# Broadcast data size and column information
metadata = [data_size, instances_columns] if rank == 0 else None
data_size, instances_columns = comm.bcast(metadata, root=0)

# Calculate counts and displacements for scatterv
counts = np.full(size, data_size // size, dtype=int) # [data_size0, data_size1,..]
remainder = data_size % size
counts[:remainder] += 1  # Add the reminder to the last process
displacements = np.insert(np.cumsum(counts[:-1]), 0, 0)

# Print information about data distribution
start_indices = displacements
end_indices = displacements + counts
print(f"Process {rank} will receive data from index {start_indices[rank]} to index {end_indices[rank] - 1}")

# Scatter data to all processes
local_data_size = counts[rank]
local_data = np.empty((local_data_size, instances_columns), dtype=float)
comm.Scatterv([data, counts * instances_columns, displacements * instances_columns, MPI.DOUBLE], local_data, root=0)

# Broadcast centroids to all processes
centroids = comm.bcast(centroids, root=0)

# Initialize local labels
local_labels = np.zeros(local_data_size, dtype=int)

start_time = time.time()  # Start execution time

for iter_num in range(MAX_ITER):
    # Each process assigns clusters to its local data
    assign_clusters(local_data, centroids, local_labels, 0, local_data_size)

    # Master process aggregates labels from all processes
    # Initialize local sums and counts
    local_sums = np.zeros((K, instances_columns)) # each process calculate the sum of atts in centroids
    local_counts = np.zeros(K, dtype=int)

    # Accumulate sums and counts for each cluster
    for i in range(local_data_size):
        local_sums[local_labels[i]] += local_data[i]
        local_counts[local_labels[i]] += 1

    # Aggregate sums and counts across all processes
    global_sums = np.zeros((K, instances_columns), dtype=float)
    global_counts = np.zeros(K, dtype=int)
    comm.Allreduce(local_sums, global_sums, op=MPI.SUM)
    comm.Allreduce(local_counts, global_counts, op=MPI.SUM)

    # Update centroids based on global sums and counts
    old_centroids = centroids[:, :-1].copy()  # Store the old centroids for comparison
    for i in range(K):
        if global_counts[i] > 0:  # Check count to avoid division by zero
            centroids[i, :-1] = global_sums[i] / global_counts[i]

    all_labels = np.zeros(data_size, dtype=int) if rank == 0 else None
    comm.Gatherv(local_labels, [all_labels, counts, displacements, MPI.INT], root=0)  # gather labels in rank 0

    # Check convergence: if centroids haven't changed, break the loop
    if rank == 0 and np.allclose(old_centroids, centroids[:, :-1]):
        print(f"Convergence reached at iteration {iter_num}")
        break



if rank == 0:
    # Measure execution time
    execution_time = time.time() - start_time
    with open('execution_time_mpi.csv', 'a') as f:
        f.write(f"{size},{execution_time}\n")

    # Print final centroids
    print("Final centroids:")
    print(centroids)

    # Print which instances belong to which centroid
    print("Instance assignments to centroids:")
    for i in range(data_size):
        print(f"Instance {i} is in cluster {all_labels[i]}")
    print(f"Convergence reached at iteration {iter_num}")
