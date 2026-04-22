import pandas as pd
import matplotlib.pyplot as plt

def analyze():
    df = pd.read_csv("data/processed/final_data.csv")

    result = df.groupby("age_group")["release_year"].mean()

    print(result)

    result.plot(kind="bar")
    plt.title("Average Release Year by Age Group")
    plt.savefig("results/figures/age_vs_year.png")
    plt.show()

if __name__ == "__main__":
    analyze()