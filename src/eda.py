import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def run_eda():
    df = pd.read_csv("data/processed/merged_data.csv")

    # 1. 발매연도 분포
    plt.figure(figsize=(12,5))
    sns.histplot(df["release_year"], bins=50, kde=True)
    plt.title("Release Year Distribution")
    plt.savefig("results/release_year_dist.png")

    # 2. 장르별 평균 연도
    genre_year = df.groupby("track_genre")["release_year"].mean().sort_values()
    genre_year.plot(kind="barh")
    plt.title("Genre vs Release Year")
    plt.savefig("results/genre_year.png")

    # 3. 상관관계
    corr = df[["release_year","popularity","music_age"]].corr()

    plt.figure(figsize=(6,4))
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.savefig("results/corr_heatmap.png")

    print("EDA 완료!")

if __name__ == "__main__":
    run_eda()