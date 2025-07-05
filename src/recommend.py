import pandas as pd
import numpy as np
import os
import joblib

def generate_recommendations():
    print("[recommend] Generating advanced recommendations...")
    # Load clustered data and models
    df = pd.read_csv(os.path.join("data", "raw", "nirf_2023_clustered.csv"))
    models_dir = os.path.join("data", "raw", "models")
    mlp_score = joblib.load(os.path.join(models_dir, "mlp_score_model.pkl"))
    mlp_rank = joblib.load(os.path.join(models_dir, "mlp_rank_model.pkl"))
    scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))
    features = ['TLR (100)', 'RPC (100)', 'GO (100)', 'OI (100)', 'PERCEPTION (100)']
    # Cluster means for benchmarking
    cluster_means = df.groupby('Cluster')[features].mean()
    recs = []
    for idx, row in df.iterrows():
        college = row['Name']
        cluster = row['Cluster']
        current = row[features]
        peer = cluster_means.loc[cluster]
        # Find weakest 2 features (relative to cluster mean)
        diffs = peer - current
        weakest = diffs.sort_values(ascending=False).head(2)
        # Simulate +10 improvement in each weak feature
        improved = current.copy()
        for feat in weakest.index:
            improved[feat] = min(improved[feat] + 10, 100)
        X_orig = np.array(current).reshape(1, -1)
        X_impr = np.array(improved).reshape(1, -1)
        X_orig_scaled = scaler.transform(X_orig)
        X_impr_scaled = scaler.transform(X_impr)
        score_orig = mlp_score.predict(X_orig_scaled)[0]
        score_impr = mlp_score.predict(X_impr_scaled)[0]
        rank_orig = mlp_rank.predict(X_orig_scaled)[0]
        rank_impr = mlp_rank.predict(X_impr_scaled)[0]
        gain = score_impr - score_orig
        rank_gain = rank_orig - rank_impr
        rec_msg = (
            f"To improve your NIRF rank, focus on: {', '.join(weakest.index)}. "
            f"Estimated score gain: {gain:.2f}, rank improvement: {rank_gain:.1f} places."
        )
        recs.append({
            'Name': college,
            'Cluster': cluster,
            'Current_Score': score_orig,
            'Current_Rank': rank_orig,
            'Weak_Feature_1': weakest.index[0],
            'Weak_Feature_2': weakest.index[1],
            'Rec_Message': rec_msg,
            'Est_Score_Gain': gain,
            'Est_Rank_Improvement': rank_gain
        })
    recs_df = pd.DataFrame(recs)
    out_path = os.path.join("data", "raw", "college_recommendations.csv")
    recs_df.to_csv(out_path, index=False)
    print(f"[recommend] Recommendations saved to {out_path}")
    print("\nSample recommendations:")
    print(recs_df[['Name', 'Rec_Message']].head(5).to_string(index=False)) 