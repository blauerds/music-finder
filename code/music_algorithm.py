import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# Load the data
music_data = pd.read_csv("rda/music_data.csv", encoding='latin1')
all_songs_data = pd.read_csv("rda/all_songs_data.csv", encoding='latin1')

# Create a list of tuples representing the (title, artist) pairs in music_data
music_pairs = list(zip(music_data["title"], music_data["artist"]))

# Filter all_songs_data to exclude rows that match the (title, artist) pairs in music_data
all_songs_data = all_songs_data.loc[~(all_songs_data[["title", "artist"]].apply(tuple, axis=1).isin(music_pairs))]

# Drop NA values
music_data = music_data.dropna()
all_songs_data = all_songs_data.dropna()

# Create a dictionary to map each unique genre to a unique number
genre_map = {}
for i, genre in enumerate(all_songs_data["top genre"]):
    genre_map[genre] = i

# Assign a unique number to each genre in the "top genre" column for both dataframes
music_data['code genre'] = music_data['top genre'].map(genre_map)
all_songs_data['code genre'] = all_songs_data['top genre'].map(genre_map)

# Get feature columns for analysis
feature_columns = ['year', 'code genre', 'bpm', 'nrgy', 'dnce', 'dB', 'val', 'dur', 'acous', 'spch']

# Separate feature columns from music_data
X_music_data = music_data[feature_columns].dropna()

# Separate feature columns from all_songs_data
X_all_songs_data = all_songs_data[feature_columns].dropna()

# Scale music data
scaler = StandardScaler()
X_music_data_scaled = scaler.fit_transform(X_music_data)

# Scale all songs data
X_all_songs_data_scaled = scaler.transform(X_all_songs_data)

# Fit KNN model to music data
knn = NearestNeighbors(n_neighbors=11)
knn.fit(X_music_data_scaled)

# Find 10 best matches in all songs data
distances, indices = knn.kneighbors(X_all_songs_data_scaled)

# Get indices of songs in all_songs_data that are not in music_data
not_in_music_data = ~all_songs_data.index.isin(music_data.index)

# Get indices of 10 best matches that are not in music_data
indices_not_in_music_data = indices[not_in_music_data][:, 1:]

# Get song titles and artists of 10 best matches
song_titles_artists = all_songs_data[['title', 'artist']].iloc[indices_not_in_music_data.flatten()].drop_duplicates()

# Get indices of songs in music_data that are not in all_songs_data
not_in_all_songs_data = ~music_data.index.isin(all_songs_data.index)

# Add songs from music_data to all_songs_data
music_data_not_in_all_songs_data = music_data[['title', 'artist']].iloc[not_in_all_songs_data].drop_duplicates()
all_songs_data = pd.concat([all_songs_data, music_data_not_in_all_songs_data], ignore_index=True)
all_songs_data.to_csv('rda/all_songs_data.csv', index=False)

print(song_titles_artists)