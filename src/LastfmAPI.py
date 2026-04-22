import requests
import json
import time

API_KEY = "YOUR_LASTFM_API_KEY"
BASE_URL = "https://ws.audioscrobbler.com/2.0/"

def get_recent_tracks(user):
    params = {
        "method": "user.getrecenttracks",
        "user": user,
        "api_key": API_KEY,
        "format": "json",
        "limit": 50
    }
    res = requests.get(BASE_URL, params=params)
    return res.json()

def collect(users):
    all_data = []

    for user in users:
        try:
            data = get_recent_tracks(user)
            all_data.append(data)
            time.sleep(0.3)
        except:
            continue

    with open("data/raw/lastfm_raw.json", "w") as f:
        json.dump(all_data, f)

if __name__ == "__main__":
    users = ["user1", "user2"]
    collect(users)