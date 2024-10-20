import time
from recommend import preprocess_data, simulate_user_preferences, collaborative_filtering

def main():
    # Start measuring execution time
    start_time = time.time()
    
    # Preprocess the data
    df = preprocess_data()
    
    # Simulate user preferences
    user_preferences = simulate_user_preferences(df)
    
    # Get collaborative filtering recommendations
    collab_recommendations = collaborative_filtering(user_preferences, df)
    
    # Print the recommendations
    print("\nSequential Collaborative Filtering Recommendations:\n", collab_recommendations)
    
    # End measuring execution time
    execution_time = time.time() - start_time
    print(f"Execution Time for sequential implementation: {execution_time:.4f} seconds")
    
    # Write the execution time to a file
    with open('sequential_execution_time.txt', 'w') as f:
        f.write(f"{execution_time:.4f}")

if __name__ == "__main__":
    main()
