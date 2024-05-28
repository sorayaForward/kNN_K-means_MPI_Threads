import matplotlib.pyplot as plt
import csv

# Lists to store data
num_processes = []
execution_times = []

# Read the execution time data from the file
with open('execution_time_mpi.csv', 'r') as file:
# with open('execution_time_th.csv', 'r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        num_processes.append(int(row[0]))  # Assuming the first column contains the number of processes
        execution_times.append(float(row[1]))  # Assuming the second column contains the execution time
print(num_processes)
# Plot the execution time against the number of processes
plt.plot(num_processes, execution_times, marker='o')
plt.xlabel('Number of Processes')
# plt.xlabel('Number of Threads')
plt.ylabel('Execution Time (seconds)')
plt.title('Execution Time vs Number of processus')
plt.show()



single_processor_time = execution_times[0]
speedup = [single_processor_time / time for time in execution_times]

# Tracer le speedup en fonction du nombre de processus
plt.plot(num_processes, speedup, marker='o')
plt.xlabel('Nombre de Processus')
# plt.xlabel('Nombre de Threads')
plt.ylabel('Accélération (Speedup)')
plt.title('Accélération vs Nombre de processus')
plt.show()
