from mpi4py import MPI
import time
from recommend import preprocess_data, simulate_user_preferences, collaborative_filtering

def main():
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Measure the start time for the execution
    start_time = time.time()

    # Assume df is your preprocessed DataFrame
    if rank == 0:
        df = preprocess_data()  # Preprocessing only done by the root process
    else:
        df = None

    # Broadcast the DataFrame to all nodes
    df = comm.bcast(df, root=0)

    # Split the user preferences among nodes
    if rank == 0:
        user_preferences = simulate_user_preferences(df)
        keys = list(user_preferences.keys())
    else:
        user_preferences = None
        keys = None

    # Broadcasting keys to all nodes
    keys = comm.bcast(keys, root=0)

    # Each node processes its subset of users
    user_subset = keys[rank::size]  # Distributing users among processes
    local_user_preferences = {k: user_preferences[k] for k in user_subset}

    recommendations = collaborative_filtering(local_user_preferences, df)

    # Gather results from all nodes
    all_recommendations = comm.gather(recommendations, root=0)

    # Measure the end time for the execution
    end_time = time.time()

    # Combine the recommendations and print results
    if rank == 0:
        combined_recommendations = {key: rec for rec_list in all_recommendations for key, rec in rec_list.items()}
        print("\nCombined Collaborative Filtering Recommendations:\n", combined_recommendations)
        
        # Print the total execution time
        execution_time = end_time - start_time
        print(f"\nTotal Execution Time: {execution_time:.4f} seconds")
        
        # Write the execution time to a file
        with open('mpi_execution_time.txt', 'w') as f:
            f.write(f"{execution_time:.4f}")

if __name__ == "__main__":
    main()
