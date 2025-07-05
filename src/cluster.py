import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

def load_data():
    data_path = os.path.join("data", "raw", "nirf_2023_cleaned.csv")
    return pd.read_csv(data_path)

def run_clustering():
    print("[cluster] Running clustering analysis...")
    df = load_data()
    features = ['TLR (100)', 'RPC (100)', 'GO (100)', 'OI (100)', 'PERCEPTION (100)']
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Find optimal number of clusters using silhouette score
    best_k = 0
    best_score = -1
    scores = []
    for k in range(2, 8):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        scores.append(score)
        if score > best_score:
            best_score = score
            best_k = k
    print(f"Best number of clusters (by silhouette score): {best_k}")
    
    # Fit final model
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # Save cluster assignments
    df.to_csv(os.path.join("data", "raw", "nirf_2023_clustered.csv"), index=False)
    
    # Cluster summary
    cluster_summary = df.groupby('Cluster')[features + ['Score', 'Rank']].mean().round(2)
    print("\nCluster Summary:")
    print(cluster_summary)
    
    # Visualize clusters (first 2 principal components)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df['PC1'] = X_pca[:, 0]
    df['PC2'] = X_pca[:, 1]
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=df, palette='Set2', s=80)
    plt.title('College Clusters (PCA Projection)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Cluster')
    plt.tight_layout()
    plt.savefig(os.path.join("data", "raw", "college_clusters_pca.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save cluster summary
    cluster_summary.to_csv(os.path.join("data", "raw", "cluster_summary.csv"))
    print("[cluster] Clustering complete! Results saved in data/raw/")

if __name__ == "__main__":
    run_clustering() 