#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define NBR_ATTRIBUTES 55 // Including the class label
#define MAX_INSTANCES 1
#define MAX_K 3
#define MAX_DATA_SIZE 581011

typedef struct {
    float attributes[NBR_ATTRIBUTES - 1]; // Without the class label
    int label;
} Instance;

typedef struct {
    float attributes[NBR_ATTRIBUTES - 1]; // Without the class label
} InstanceC;

typedef struct {
    int label; // Class label
    float distance; // Distance to the instance
} NeighborInfo;

typedef struct {
    Instance* data;
    InstanceC* instance;
    NeighborInfo* results;
    int start_index;
    int end_index;
} ThreadData;

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
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error opening file %s\n", filename);
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < MAX_DATA_SIZE; i++) {
        for (int j = 0; j < NBR_ATTRIBUTES - 1; j++) { // read all attributes
            if (fscanf(file, "%f,", &data[i].attributes[j]) != 1) {
                fprintf(stderr, "Error reading attribute %d of instance %d\n", j, i);
                exit(EXIT_FAILURE);
            }
        }
        if (fscanf(file, "%d\n", &data[i].label) != 1) { // read class label
            fprintf(stderr, "Error reading label of instance %d\n", i);
            exit(EXIT_FAILURE);
        }
    }
    fclose(file);
}

void load_instances(InstanceC* instances, const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error opening file %s\n", filename);
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < MAX_INSTANCES; i++) {
        for (int j = 0; j < NBR_ATTRIBUTES - 1; j++) {
            if (fscanf(file, "%f,", &instances[i].attributes[j]) != 1) {
                fprintf(stderr, "Error reading attribute %d of instance %d\n", j, i);
                exit(EXIT_FAILURE);
            }
        }
    }
    fclose(file);
}

float manhattan_distance(const Instance* instance1, const InstanceC* instance2) {
    float distance = 0.0;
    for (int i = 0; i < NBR_ATTRIBUTES - 1; i++) {
        distance += fabs(instance1->attributes[i] - instance2->attributes[i]);
    }
    return distance;
}

void* knn(void* arg) {
    printf("Entering knn function\n"); // Debugging statement
    ThreadData* thread_data = (ThreadData*)arg;
    Instance* data_chunk = thread_data->data;
    InstanceC* instance = thread_data->instance;
    NeighborInfo* knn_results = thread_data->results;
    int start_index = thread_data->start_index;
    int end_index = thread_data->end_index;

    float* distances = (float*)malloc(MAX_DATA_SIZE * sizeof(float));
    if (!distances) {
        fprintf(stderr, "Memory allocation error for distances array\n");
        pthread_exit(NULL);
    }

    for (int i = start_index; i < end_index; i++) {
        distances[i] = manhattan_distance(&data_chunk[i], instance);
    }

    for (int j = 0; j < MAX_K; j++) {
        int min_index = start_index;
        for (int l = start_index + 1; l < end_index; l++) {
            if (distances[l] < distances[min_index]) {
                min_index = l;
            }
        }
        knn_results[j].label = data_chunk[min_index].label;
        knn_results[j].distance = distances[min_index];
        distances[min_index] = INFINITY;
    }

    free(distances);
    printf("Exiting knn function\n"); // Debugging statement
    pthread_exit(NULL);
}

int main(int argc, char** argv) {
    int NUM_THREADS = 0;  // Number of threads to use
    int z = 25;
    while (NUM_THREADS <= z) {
        NUM_THREADS = NUM_THREADS + 1;
        Instance* data = (Instance*)malloc(MAX_DATA_SIZE * sizeof(Instance));
        InstanceC* instances = (InstanceC*)malloc(MAX_INSTANCES * sizeof(InstanceC));
        NeighborInfo* knn_results = (NeighborInfo*)malloc(MAX_K * NUM_THREADS * sizeof(NeighborInfo));
        pthread_t* threads = (pthread_t*)malloc(NUM_THREADS * sizeof(pthread_t));
        ThreadData* thread_data = (ThreadData*)malloc(NUM_THREADS * sizeof(ThreadData));
        clock_t start, end;
        double cpu_time_used;

        if (!data || !instances || !knn_results || !threads || !thread_data) {
            fprintf(stderr, "Memory allocation error\n");
            exit(EXIT_FAILURE);
        }

        // Load data before creating the threads
        load_data(data, "./dataset/covtype.csv");
        load_instances(instances, "./dataset/instances.csv");

        int chunk_size = (MAX_DATA_SIZE + NUM_THREADS - 1) / NUM_THREADS;

        start = clock();

        for (int i = 0; i < NUM_THREADS; i++) {
            int start_index = i * chunk_size;
            int end_index = (i == NUM_THREADS - 1) ? MAX_DATA_SIZE : (i + 1) * chunk_size;
            thread_data[i] = (ThreadData){ data, instances, knn_results + i * MAX_K, start_index, end_index };
            if (pthread_create(&threads[i], NULL, knn, &thread_data[i]) != 0) {
                fprintf(stderr, "Error creating thread %d\n", i);
                exit(EXIT_FAILURE);
            }
        }

        for (int i = 0; i < NUM_THREADS; i++) {
            if (pthread_join(threads[i], NULL) != 0) {
                fprintf(stderr, "Error joining thread %d\n", i);
                exit(EXIT_FAILURE);
            }
        }

        end = clock();
        cpu_time_used = end - start;

        printf("Execution Time: %f seconds\n", cpu_time_used);

        sort_neighbors(knn_results, MAX_K * NUM_THREADS);

        printf("K Nearest Neighbors are:\n");
        for (int i = 0; i < MAX_K; i++) {
            printf("Distance: %f, Label: %d\n", knn_results[i].distance, knn_results[i].label);
        }

        FILE* file = fopen("execution_time.csv", "a");
        if (file) {
            fprintf(file, "%d,%f\n", NUM_THREADS, cpu_time_used);
            fclose(file);
        } else {
            printf("Error opening execution_time.csv for writing\n");
        }

        // Free allocated memory
        free(data);
        free(instances);
        free(knn_results);
        free(threads);
        free(thread_data);
    }
    return 0;
}
