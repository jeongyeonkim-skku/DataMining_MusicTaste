import pandas as pd

def preprocess():
    spotify = pd.read_csv("data/SpotifyFeatures.csv")
    survey = pd.read_csv("data/survey.csv")

    # Spotify: 날짜 → 연도
    spotify["release_year"] = pd.to_datetime(
        spotify["release_date"], errors="coerce"
    ).dt.year

    spotify = spotify[["track_name", "artists", "release_year"]]

    # 컬럼 이름 맞추기
    survey.columns = ["age", "track_name", "artist"]

    # merge
    df = pd.merge(
        survey,
        spotify,
        left_on="track_name",
        right_on="track_name",
        how="left"
    )

    # 결측 제거
    df = df.dropna(subset=["release_year"])

    # 음악 나이
    df["music_age"] = 2026 - df["release_year"]

    # 연령대
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 20, 30, 40, 50, 100],
        labels=["10s","20s","30s","40s","50+"]
    )

    df.to_csv("data/processed/final_data.csv", index=False)
    print("Done!")

if __name__ == "__main__":
    preprocess()