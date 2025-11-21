import spotipy
from spotipy.oauth2 import SpotifyOAuth

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id="b87f1c7848554a2cad9f133ea31b7b68",
    client_secret="ead04af1027b460890f0b9caee1bd61e",
    redirect_uri="http://127.0.0.1:8080/callback",
    scope="user-library-read playlist-read-private playlist-read-collaborative"
))

results = sp.search(q="happy playlist", type="playlist", limit=1)
print(results)
