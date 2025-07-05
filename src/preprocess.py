import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

def clean_nirf_data():
    input_path = os.path.join("data", "raw", "nirf_2023.csv")
    output_path = os.path.join("data", "raw", "nirf_2023_cleaned.csv")
    print(f"[preprocess] Loading {input_path}")
    df = pd.read_csv(input_path)
    print(f"[preprocess] Initial shape: {df.shape}")
    print(f"[preprocess] Columns: {df.columns.tolist()}")
    # Drop duplicates
    df = df.drop_duplicates()
    # Handle missing values (drop rows with any missing for now)
    df = df.dropna()
    print(f"[preprocess] After cleaning: {df.shape}")
    df.to_csv(output_path, index=False)
    print(f"[preprocess] Cleaned data saved to {output_path}")

def clean_data():
    print("[preprocess] Cleaning and preprocessing data...")
    clean_nirf_data()

def eda():
    print("[preprocess] Running EDA...")
    data_path = os.path.join("data", "raw", "nirf_2023_cleaned.csv")
    df = pd.read_csv(data_path)
    print(df.describe())
    # Top 10 colleges by rank
    top10 = df.sort_values("Rank").head(10)
    plt.figure(figsize=(10,6))
    sns.barplot(y="Name", x="Score", data=top10, palette="viridis")
    plt.title("Top 10 Colleges by NIRF Rank (2023)")
    plt.xlabel("Score")
    plt.ylabel("College Name")
    plt.tight_layout()
    plt.savefig(os.path.join("data", "raw", "top10_colleges.png"))
    plt.close()
    # Feature distributions
    features = ["Score", "TLR (100)", "RPC (100)", "GO (100)", "OI (100)", "PERCEPTION (100)"]
    for feat in features:
        plt.figure(figsize=(8,4))
        sns.histplot(df[feat], kde=True, bins=20, color="skyblue")
        plt.title(f"Distribution of {feat}")
        plt.xlabel(feat)
        plt.tight_layout()
        plt.savefig(os.path.join("data", "raw", f"dist_{feat.replace(' ', '_').replace('(', '').replace(')', '')}.png"))
        plt.close()
    # Correlation heatmap
    plt.figure(figsize=(8,6))
    corr = df[features].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap (NIRF 2023)")
    plt.tight_layout()
    plt.savefig(os.path.join("data", "raw", "correlation_heatmap.png"))
    plt.close()
    print("[preprocess] EDA plots saved in data/raw/") 