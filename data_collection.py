import requests
import pandas as pd

# Spotify API credentials
CLIENT_ID = '98c10be9aa874577b09e17f368adc2f0'
CLIENT_SECRET = '57df31960a274e9ca4e6f793d763b276'

def get_access_token(client_id, client_secret):
    url = 'https://accounts.spotify.com/api/token'
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    data = {'grant_type': 'client_credentials'}

    response = requests.post(url, headers=headers, data=data, auth=(client_id, client_secret))
    token = response.json().get('access_token')
    return token

def get_track_audio_features(track_id, access_token):
    url = f"https://api.spotify.com/v1/audio-features/{track_id}"
    headers = {'Authorization': f'Bearer {access_token}'}
    response = requests.get(url, headers=headers)
    return response.json()

def get_playlist_tracks(playlist_id, access_token):
    url = f"https://api.spotify.com/v1/playlists/{playlist_id}/tracks"
    headers = {'Authorization': f'Bearer {access_token}'}
    response = requests.get(url, headers=headers)
    data = response.json()
    
    # Return track info as well
    track_data = []
    for track in data['items']:
        track_info = {
            'id': track['track']['id'],
            'name': track['track']['name'],
            'artists': ', '.join([artist['name'] for artist in track['track']['artists']]),
        }
        track_data.append(track_info)

    return track_data

def collect_data():
    access_token = get_access_token(CLIENT_ID, CLIENT_SECRET)
    playlist_id = '2nWAZcqs4owoTcIpKx4lep'  # Replace with your own playlist ID
    track_data = get_playlist_tracks(playlist_id, access_token)
    
    # Fetch audio features for each track
    for track in track_data:
        features = get_track_audio_features(track['id'], access_token)
        track.update(features)  # Combine track info with its audio features

    df = pd.DataFrame(track_data)
    df.to_csv('spotify_track_data.csv', index=False)
    print("Track data saved to spotify_track_data.csv")
    return df

if __name__ == "__main__":
    collect_data()
