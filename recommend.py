import pandas as pd
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

def preprocess_data():
    df = pd.read_csv('spotify_track_data.csv')
    scaler = MinMaxScaler()
    df[['tempo', 'energy', 'danceability']] = scaler.fit_transform(df[['tempo', 'energy', 'danceability']])
    return df

def simulate_user_preferences(df, n_users=10):
    user_preferences = {}
    for user in range(n_users):
        # Randomly select 5 songs for each user
        song_indices = random.sample(range(len(df)), 5)
        user_preferences[f"User {user + 1}"] = df.iloc[song_indices]['name'].tolist()
    return user_preferences

def collaborative_filtering(user_preferences, df):
    recommendations = {}
    features = ['tempo', 'energy', 'danceability']
    similarity_matrix = cosine_similarity(df[features])
    
    for user, liked_songs in user_preferences.items():
        # Get indices of liked songs
        liked_indices = df[df['name'].isin(liked_songs)].index.tolist()
        
        # Calculate similarity scores for liked songs
        similar_scores = {}
        for idx in liked_indices:
            for i, score in enumerate(similarity_matrix[idx]):
                if df.iloc[i]['name'] not in liked_songs:
                    if df.iloc[i]['name'] not in similar_scores:
                        similar_scores[df.iloc[i]['name']] = score
                    else:
                        similar_scores[df.iloc[i]['name']] += score
        
        # Sort by scores and recommend top 5 songs
        sorted_recommendations = sorted(similar_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        recommendations[user] = [song[0] for song in sorted_recommendations]
    
    return recommendations

def content_based_filtering(track_name, df, n_recommendations=5):
    features = ['tempo', 'energy', 'danceability']
    track_idx = df[df['name'] == track_name].index[0]
    similarity_matrix = cosine_similarity(df[features])
    
    similarity_scores = list(enumerate(similarity_matrix[track_idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    
    top_recommendations = [i[0] for i in similarity_scores[1:n_recommendations + 1]]
    return df.iloc[top_recommendations][['name', 'artists']]

if __name__ == "__main__":
    df = preprocess_data()  # Ensure your data is preprocessed
    user_preferences = simulate_user_preferences(df)
    
    print("Simulated User Preferences:\n", user_preferences)
    
    # Get collaborative filtering recommendations
    collab_recommendations = collaborative_filtering(user_preferences, df)
    print("\nCollaborative Filtering Recommendations:\n", collab_recommendations)
    
    # Example for content-based filtering
    example_track = random.choice(df['name'].tolist())
    content_recommendations = content_based_filtering(example_track, df)
    print("\nContent-Based Filtering Recommendations for '{}':\n".format(example_track), content_recommendations)
