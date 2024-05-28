#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define NBR_ATTRIBUTES 55
#define MAX_INSTANCES 1
#define MAX_K 3
#define MAX_DATA_SIZE 581011
#define MAX_PROCESSES 20

typedef struct {
    float attributes[NBR_ATTRIBUTES - 1];
    int label;
} Instance;

typedef struct {
    float attributes[NBR_ATTRIBUTES - 1];
} InstanceC;

typedef struct {
    int start_index;
    int end_index;
    Instance* instances;
} DataChunk;

typedef struct {
    int label;
    double distance;
} NeighborInfo;

void sort_neighbors(NeighborInfo* neighbors, int n) {
    for (int i = 0; i < n - 1; i++) {
        int min_index = i;
        for (int j = i + 1; j < n; j++) {
            if (neighbors[j].distance < neighbors[min_index].distance) {
                min_index = j;
            }
        }
        NeighborInfo temp = neighbors[i];
        neighbors[i] = neighbors[min_index];
        neighbors[min_index] = temp;
    }
}

void load_data(Instance* data, const char* filename) {
    FILE* file;
    if (fopen_s(&file, filename, "r") != 0) {
        fprintf(stderr, "Error opening file %s\n", filename);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    for (int i = 0; i < MAX_DATA_SIZE; i++) {
        for (int j = 0; j < NBR_ATTRIBUTES - 1; j++) {
            if (fscanf_s(file, "%f,", &data[i].attributes[j]) != 1) {
                if (feof(file)) {
                    fprintf(stderr, "End of file reached unexpectedly at instance %d, attribute %d\n", i, j);
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                else {
                    fprintf(stderr, "Error reading attribute %d from instance %d in file %s\n", j, i, filename);
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
            }
        }
        if (fscanf_s(file, "%d\n", &data[i].label) != 1) {
            fprintf(stderr, "Error reading label for instance %d in file %s\n", i, filename);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    fclose(file);
}

void load_instances(InstanceC* instances, const char* filename) {
    FILE* file;
    if (fopen_s(&file, filename, "r") != 0) {
        fprintf(stderr, "Error opening file %s\n", filename);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    for (int i = 0; i < MAX_INSTANCES; i++) {
        for (int j = 0; j < NBR_ATTRIBUTES - 1; j++) {
            if (fscanf_s(file, "%f,", &instances[i].attributes[j]) != 1) {
                fprintf(stderr, "Error reading attribute %d from instance %d in file %s\n", j, i, filename);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
    }
    fclose(file);
}

double manhattan_distance(Instance* instance1, InstanceC* instance2) {
    double distance = 0.0;
    for (int i = 0; i < NBR_ATTRIBUTES - 1; i++) {
        distance += fabs(instance1->attributes[i] - instance2->attributes[i]);
    }
    return distance;
}

void knn(DataChunk* data_chunk, InstanceC* instances, NeighborInfo* knn_results) {
    int lar = data_chunk->end_index - data_chunk->start_index;
    float* distances = (float*)malloc(lar * sizeof(float));

    if (distances == NULL) {
        fprintf(stderr, "Memory allocation failed for distances array\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (int i = 0; i < lar; i++) {
        distances[i] = manhattan_distance(&data_chunk->instances[i], &instances[0]);
    }

    for (int j = 0; j < MAX_K; j++) {
        int min_index = 0;
        for (int l = 1; l < lar; l++) {
            if (distances[l] < distances[min_index]) {
                min_index = l;
            }
        }

        knn_results[j].label = data_chunk->instances[min_index].label;
        knn_results[j].distance = distances[min_index];
        distances[min_index] = INFINITY;
    }
    free(distances);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    Instance* data = (Instance*)malloc(MAX_DATA_SIZE * sizeof(Instance));
    InstanceC* instances = (InstanceC*)malloc(MAX_INSTANCES * sizeof(InstanceC));
    NeighborInfo* knn_results = (NeighborInfo*)malloc(MAX_K * sizeof(NeighborInfo));

    clock_t start, end;
    double cpu_time_used;

    if (world_rank == 0) {
        load_data(data, "covtype.csv");
        load_instances(instances, "instances.csv");
        start = clock();
    }

    if (world_size > 1) { // bcast only when nbr of processes > 1
        MPI_Bcast(instances, MAX_INSTANCES * sizeof(InstanceC), MPI_BYTE, 0, MPI_COMM_WORLD);
    }
    DataChunk chunk;
    chunk.instances = (Instance*)malloc((MAX_DATA_SIZE / world_size + 1) * sizeof(Instance));

    if (chunk.instances == NULL) {
        fprintf(stderr, "Memory allocation failed for chunk.instances array\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int data_per_process = MAX_DATA_SIZE / world_size;
    int remaining_data = MAX_DATA_SIZE % world_size;

    // Calculate start and end indices for each process
    chunk.start_index = world_rank * data_per_process + (remaining_data > world_rank ? world_rank : remaining_data);
    chunk.end_index = (world_rank + 1) * data_per_process + (remaining_data > (world_rank + 1) ? (world_rank + 1) : remaining_data);

    if (world_rank == 0) {
        // Process 0's chunk
        printf(">>> Process %d: processes data from index %d to %d\n", world_rank, chunk.start_index, chunk.end_index - 1); // Debugging statement
        for (int j = chunk.start_index; j < chunk.end_index; j++) {
            memcpy(&chunk.instances[j - chunk.start_index], &data[j], sizeof(Instance));
        }
        knn(&chunk, instances, knn_results);
        printf("Process %d: Calculated k nearest neighbors\n", world_rank); // Debugging statement
        for (int i = 0; i < MAX_K; i++) {
            printf("%f %d\n", knn_results[i].distance, knn_results[i].label); // Debugging statement
        }
        // Send data to worker processes
        for (int i = 1; i < world_size; i++) {
            int start_index = i * data_per_process + (remaining_data > i ? i : remaining_data);
            int end_index = (i + 1) * data_per_process + (remaining_data > (i + 1) ? (i + 1) : remaining_data);
            int chunk_size = end_index - start_index;

            MPI_Send(&data[start_index], chunk_size * sizeof(Instance), MPI_BYTE, i, 0, MPI_COMM_WORLD);
        }
    }
    else {
        int chunk_size = chunk.end_index - chunk.start_index;
        MPI_Recv(chunk.instances, chunk_size * sizeof(Instance), MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        printf(">>> Process %d: Received data from index %d to %d\n", world_rank, chunk.start_index, chunk.end_index - 1); // Debugging statement
        knn(&chunk, instances, knn_results);
        printf("Process %d: Calculated k nearest neighbors\n", world_rank); // Debugging statement
        for (int i = 0; i < MAX_K; i++) {
            printf("%f %d\n", knn_results[i].distance, knn_results[i].label); // Debugging statement
        }
    }

    NeighborInfo* recv_knn_results = NULL;
    if (world_rank == 0) {
        recv_knn_results = (NeighborInfo*)malloc(world_size * MAX_K * sizeof(NeighborInfo));
    }
    if (world_size > 1) { // if more than one process then gather results in process 0
        MPI_Gather(knn_results, MAX_K * sizeof(NeighborInfo), MPI_BYTE, recv_knn_results, MAX_K * sizeof(NeighborInfo), MPI_BYTE, 0, MPI_COMM_WORLD);
    }
    if (world_rank == 0) {
        end = clock();
        sort_neighbors(recv_knn_results, MAX_K * world_size);

        printf("************************************\n");
        printf("K Nearest Neighbors are: \n");

        for (int j = 0; j < MAX_K; j++) {
            printf("%f %d\n", (world_size == 1 ? knn_results[j].distance : recv_knn_results[j].distance), (world_size == 1 ? knn_results[j].label : recv_knn_results[j].label));
        }
        printf("************************************\n");

        FILE* fp;
        errno_t err = fopen_s(&fp, "...../execution_time.csv", "a");
        if (err != 0) {
            printf("Error opening file: %d\n", err);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        cpu_time_used = end - start;
        fprintf(fp, "%d, %f\n", world_size, cpu_time_used);
        fclose(fp);
    }

    free(knn_results);
    free(data);
    free(instances);
    free(chunk.instances);
    if (world_rank == 0) {
        free(recv_knn_results);
    }

    MPI_Finalize();

    return 0;
}
