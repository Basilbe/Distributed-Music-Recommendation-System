import matplotlib.pyplot as plt
import numpy as np

# Read execution times from files
with open('sequential_execution_time.txt', 'r') as seq_file:
    sequential_time = float(seq_file.read())

with open('mpi_execution_time.txt', 'r') as mpi_file:
    mpi_time = float(mpi_file.read())

# Data
implementations = ['Sequential', 'MPI']
execution_times = [sequential_time, mpi_time]

# Bar positions
x = np.arange(len(implementations))

# Create grouped bar chart
plt.bar(x, execution_times, color=['blue', 'orange'], width=0.4)
plt.xticks(x, implementations)
plt.ylabel('Execution Time (seconds)')
plt.title('Execution Time Comparison')
plt.ylim(0, max(execution_times) + 0.1)

# Adding data labels
for i, v in enumerate(execution_times):
    plt.text(i, v + 0.01, f"{v:.4f}", ha='center', color='black')

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
