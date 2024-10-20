import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

def preprocess_data():
    df = pd.read_csv('spotify_track_data.csv')
    scaler = MinMaxScaler()
    df[['tempo', 'energy', 'danceability']] = scaler.fit_transform(df[['tempo', 'energy', 'danceability']])
    df.to_csv('preprocessed_spotify_data.csv', index=False)
    print("Preprocessed data saved to preprocessed_spotify_data.csv")
    return df

def recommend_tracks(track_name, df, n_recommendations=5):
    features = ['tempo', 'energy', 'danceability']
    track_idx = df[df['name'] == track_name].index[0]
    similarity_matrix = cosine_similarity(df[features])
    similarity_scores = list(enumerate(similarity_matrix[track_idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    top_recommendations = [i[0] for i in similarity_scores[1:n_recommendations + 1]]
    return df.iloc[top_recommendations][['name', 'artists']]  # Changed from 'track_name' to 'name'

if __name__ == "__main__":
    df = preprocess_data()  # Preprocess the collected data
    track_name = input("Enter a track name from the dataset: ")
    recommended_tracks = recommend_tracks(track_name, df)
    print("\nRecommended Tracks:\n", recommended_tracks)
