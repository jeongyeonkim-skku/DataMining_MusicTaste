import requests
import pandas as pd

CLIENT_ID = "YOUR_CLIENT_ID"
CLIENT_SECRET = "YOUR_CLIENT_SECRET"

# 토큰 받기
def get_token():
    url = "https://accounts.spotify.com/api/token"
    data = {"grant_type": "client_credentials"}
    response = requests.post(url, data=data, auth=(CLIENT_ID, CLIENT_SECRET))
    return response.json()["access_token"]

# 트랙 검색
def get_tracks(query="kpop", limit=50):
    token = get_token()
    headers = {"Authorization": f"Bearer {token}"}

    url = "https://api.spotify.com/v1/search"
    params = {
        "q": query,
        "type": "track",
        "limit": limit
    }

    res = requests.get(url, headers=headers, params=params)
    data = res.json()

    tracks = []
    for item in data["tracks"]["items"]:
        tracks.append({
            "track_name": item["name"],
            "artist": item["artists"][0]["name"],
            "release_date": item["album"]["release_date"],
            "popularity": item["popularity"]
        })

    return pd.DataFrame(tracks)

# 실행
df = get_tracks("kpop", 50)
df.to_csv("spotify_data.csv", index=False)

print("CSV 저장 완료")