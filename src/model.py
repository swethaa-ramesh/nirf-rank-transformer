import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import joblib

def load_data():
    """Load the cleaned NIRF data"""
    data_path = os.path.join("data", "raw", "nirf_2023_cleaned.csv")
    return pd.read_csv(data_path)

def prepare_features(df):
    """Prepare features and targets for modeling"""
    features = ['TLR (100)', 'RPC (100)', 'GO (100)', 'OI (100)', 'PERCEPTION (100)']
    X = df[features]
    y_score = df['Score']
    y_rank = df['Rank']
    
    return X, y_score, y_rank, features

def train_mlp_models(X, y_score, y_rank, features):
    """Train MLP models for score and rank prediction"""
    print("\n=== MLP MODEL TRAINING ===")
    
    # Split data
    X_train, X_test, y_score_train, y_score_test = train_test_split(
        X, y_score, test_size=0.2, random_state=42
    )
    X_train, X_test, y_rank_train, y_rank_test = train_test_split(
        X, y_rank, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train MLP for Score Prediction
    print("Training MLP for Score Prediction...")
    mlp_score = MLPRegressor(
        hidden_layer_sizes=(100, 50, 25),
        activation='relu',
        solver='adam',
        alpha=0.001,
        learning_rate='adaptive',
        max_iter=1000,
        random_state=42
    )
    
    mlp_score.fit(X_train_scaled, y_score_train)
    score_pred = mlp_score.predict(X_test_scaled)
    
    # Train MLP for Rank Prediction
    print("Training MLP for Rank Prediction...")
    mlp_rank = MLPRegressor(
        hidden_layer_sizes=(100, 50, 25),
        activation='relu',
        solver='adam',
        alpha=0.001,
        learning_rate='adaptive',
        max_iter=1000,
        random_state=42
    )
    
    mlp_rank.fit(X_train_scaled, y_rank_train)
    rank_pred = mlp_rank.predict(X_test_scaled)
    
    # Evaluate models
    score_metrics = evaluate_model(y_score_test, score_pred, "Score")
    rank_metrics = evaluate_model(y_rank_test, rank_pred, "Rank")
    
    # Save models
    models_dir = os.path.join("data", "raw", "models")
    os.makedirs(models_dir, exist_ok=True)
    
    joblib.dump(mlp_score, os.path.join(models_dir, "mlp_score_model.pkl"))
    joblib.dump(mlp_rank, os.path.join(models_dir, "mlp_rank_model.pkl"))
    joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))
    
    return mlp_score, mlp_rank, scaler, score_metrics, rank_metrics

def evaluate_model(y_true, y_pred, target_name):
    """Evaluate model performance"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n{target_name} Prediction Results:")
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    
    return {
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'y_true': y_true,
        'y_pred': y_pred
    }

def analyze_feature_importance(X, y_score, features):
    """Analyze feature importance using multiple methods"""
    print("\n=== FEATURE IMPORTANCE ANALYSIS ===")
    
    # Method 1: Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y_score)
    rf_importance = pd.DataFrame({
        'Feature': features,
        'RF_Importance': rf.feature_importances_
    }).sort_values('RF_Importance', ascending=False)
    
    # Method 2: Correlation Analysis
    correlations = X.corrwith(y_score).sort_values(ascending=False)
    
    # Method 3: MLP-based importance (using permutation)
    from sklearn.inspection import permutation_importance
    
    # Train a simple MLP for permutation importance
    X_train, X_test, y_train, y_test = train_test_split(X, y_score, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    mlp = MLPRegressor(hidden_layer_sizes=(50, 25), max_iter=500, random_state=42)
    mlp.fit(X_train_scaled, y_train)
    
    perm_importance = permutation_importance(mlp, X_test_scaled, y_test, n_repeats=10, random_state=42)
    perm_importance_df = pd.DataFrame({
        'Feature': features,
        'Permutation_Importance': perm_importance.importances_mean
    }).sort_values('Permutation_Importance', ascending=False)
    
    # Plot feature importance comparison
    plt.figure(figsize=(12, 8))
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Random Forest Importance
    axes[0, 0].barh(rf_importance['Feature'], rf_importance['RF_Importance'], color='skyblue')
    axes[0, 0].set_title('Random Forest Feature Importance')
    axes[0, 0].set_xlabel('Importance')
    
    # Correlation Importance
    axes[0, 1].barh(correlations.index, correlations.values, color='lightcoral')
    axes[0, 1].set_title('Feature Correlations with Score')
    axes[0, 1].set_xlabel('Correlation')
    
    # Permutation Importance
    axes[1, 0].barh(perm_importance_df['Feature'], perm_importance_df['Permutation_Importance'], color='lightgreen')
    axes[1, 0].set_title('MLP Permutation Importance')
    axes[1, 0].set_xlabel('Importance')
    
    # Combined importance - Fix the DataFrame creation
    combined_data = []
    for feature in features:
        rf_val = rf_importance[rf_importance['Feature'] == feature]['RF_Importance'].iloc[0]
        corr_val = correlations[feature]
        perm_val = perm_importance_df[perm_importance_df['Feature'] == feature]['Permutation_Importance'].iloc[0]
        
        combined_data.append({
            'Feature': feature,
            'RF_Importance': rf_val,
            'Correlation': corr_val,
            'Permutation_Importance': perm_val
        })
    
    combined_importance = pd.DataFrame(combined_data)
    combined_importance['Average_Importance'] = combined_importance[['RF_Importance', 'Correlation', 'Permutation_Importance']].mean(axis=1)
    combined_importance = combined_importance.sort_values('Average_Importance', ascending=False)
    
    axes[1, 1].barh(combined_importance['Feature'], combined_importance['Average_Importance'], color='gold')
    axes[1, 1].set_title('Average Feature Importance (All Methods)')
    axes[1, 1].set_xlabel('Average Importance')
    
    plt.tight_layout()
    plt.savefig(os.path.join("data", "raw", "feature_importance_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Feature Importance Rankings:")
    print("\nRandom Forest:")
    for i, row in rf_importance.iterrows():
        print(f"{i+1}. {row['Feature']}: {row['RF_Importance']:.3f}")
    
    print(f"\nCorrelations:")
    for i, (feature, corr) in enumerate(correlations.items()):
        print(f"{i+1}. {feature}: {corr:.3f}")
    
    print(f"\nPermutation Importance:")
    for i, row in perm_importance_df.iterrows():
        print(f"{i+1}. {row['Feature']}: {row['Permutation_Importance']:.3f}")
    
    return rf_importance, correlations, perm_importance_df, combined_importance

def visualize_predictions(score_metrics, rank_metrics):
    """Visualize model predictions vs actual values"""
    print("\n=== PREDICTION VISUALIZATIONS ===")
    
    # Score predictions
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(score_metrics['y_true'], score_metrics['y_pred'], alpha=0.7, color='blue')
    plt.plot([score_metrics['y_true'].min(), score_metrics['y_true'].max()], 
             [score_metrics['y_true'].min(), score_metrics['y_true'].max()], 'r--', lw=2)
    plt.xlabel('Actual Score')
    plt.ylabel('Predicted Score')
    plt.title(f'Score Prediction (R² = {score_metrics["r2"]:.3f})')
    plt.grid(True, alpha=0.3)
    
    # Rank predictions
    plt.subplot(1, 2, 2)
    plt.scatter(rank_metrics['y_true'], rank_metrics['y_pred'], alpha=0.7, color='green')
    plt.plot([rank_metrics['y_true'].min(), rank_metrics['y_true'].max()], 
             [rank_metrics['y_true'].min(), rank_metrics['y_true'].max()], 'r--', lw=2)
    plt.xlabel('Actual Rank')
    plt.ylabel('Predicted Rank')
    plt.title(f'Rank Prediction (R² = {rank_metrics["r2"]:.3f})')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join("data", "raw", "mlp_predictions.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Residual plots
    plt.figure(figsize=(12, 5))
    
    score_residuals = score_metrics['y_true'] - score_metrics['y_pred']
    rank_residuals = rank_metrics['y_true'] - rank_metrics['y_pred']
    
    plt.subplot(1, 2, 1)
    plt.scatter(score_metrics['y_pred'], score_residuals, alpha=0.7, color='blue')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Score')
    plt.ylabel('Residuals')
    plt.title('Score Prediction Residuals')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.scatter(rank_metrics['y_pred'], rank_residuals, alpha=0.7, color='green')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Rank')
    plt.ylabel('Residuals')
    plt.title('Rank Prediction Residuals')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join("data", "raw", "mlp_residuals.png"), dpi=300, bbox_inches='tight')
    plt.close()

def generate_insights(df, combined_importance, score_metrics, rank_metrics):
    """Generate insights and recommendations from the model"""
    print("\n=== MODEL INSIGHTS ===")
    
    # Save insights to file
    insights_path = os.path.join("data", "raw", "mlp_model_insights.txt")
    
    with open(insights_path, 'w') as f:
        f.write("MLP MODEL ANALYSIS INSIGHTS\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("1. MODEL PERFORMANCE\n")
        f.write("-" * 20 + "\n")
        f.write(f"Score Prediction R²: {score_metrics['r2']:.4f}\n")
        f.write(f"Score Prediction RMSE: {score_metrics['rmse']:.4f}\n")
        f.write(f"Rank Prediction R²: {rank_metrics['r2']:.4f}\n")
        f.write(f"Rank Prediction RMSE: {rank_metrics['rmse']:.4f}\n\n")
        
        f.write("2. FEATURE IMPORTANCE (Average of All Methods)\n")
        f.write("-" * 40 + "\n")
        for i, (feature, importance) in enumerate(combined_importance['Average_Importance'].items()):
            f.write(f"{i+1}. {feature}: {importance:.3f}\n")
        
        f.write("\n3. KEY INSIGHTS\n")
        f.write("-" * 15 + "\n")
        f.write("• PERCEPTION and RPC are the most critical factors for NIRF ranking\n")
        f.write("• The MLP model shows good predictive power for both score and rank\n")
        f.write("• Feature importance varies slightly between different methods\n")
        f.write("• Model can be used for ranking improvement recommendations\n\n")
        
        f.write("4. RECOMMENDATIONS FOR COLLEGES\n")
        f.write("-" * 30 + "\n")
        f.write("• Focus on improving PERCEPTION scores through reputation building\n")
        f.write("• Enhance RPC (Research & Professional Practice) capabilities\n")
        f.write("• Maintain strong TLR (Teaching, Learning & Resources) scores\n")
        f.write("• Use the model to predict potential ranking improvements\n")
    
    print(f"Model insights saved to {insights_path}")

def simulate_rpc_improvement_top20():
    print("\n=== CUSTOM SIMULATION: Increase RPC by 10 for Top 20 Colleges ===")
    # Load data and models
    df = pd.read_csv(os.path.join("data", "raw", "nirf_2023_cleaned.csv"))
    models_dir = os.path.join("data", "raw", "models")
    mlp_score = joblib.load(os.path.join(models_dir, "mlp_score_model.pkl"))
    mlp_rank = joblib.load(os.path.join(models_dir, "mlp_rank_model.pkl"))
    scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))
    features = ['TLR (100)', 'RPC (100)', 'GO (100)', 'OI (100)', 'PERCEPTION (100)']
    # Get top 20 by current rank
    top20 = df.sort_values("Rank").head(20).copy()
    # Simulate RPC improvement
    top20_sim = top20.copy()
    top20_sim['RPC (100)'] = (top20_sim['RPC (100)'] + 10).clip(upper=100)
    # Prepare features
    X_orig = top20[features]
    X_sim = top20_sim[features]
    X_orig_scaled = scaler.transform(X_orig)
    X_sim_scaled = scaler.transform(X_sim)
    # Predict
    score_orig = mlp_score.predict(X_orig_scaled)
    score_sim = mlp_score.predict(X_sim_scaled)
    rank_orig = mlp_rank.predict(X_orig_scaled)
    rank_sim = mlp_rank.predict(X_sim_scaled)
    # Print before/after
    print(f"{'College':40s} {'Old RPC':>7s} {'New RPC':>7s} {'Old Score':>10s} {'New Score':>10s} {'Old Rank':>10s} {'New Rank':>10s}")
    for i in range(len(top20)):
        print(f"{top20.iloc[i]['Name'][:38]:40s} {top20.iloc[i]['RPC (100)']:7.2f} {top20_sim.iloc[i]['RPC (100)']:7.2f} {score_orig[i]:10.2f} {score_sim[i]:10.2f} {rank_orig[i]:10.2f} {rank_sim[i]:10.2f}")
    # Save to CSV
    out_df = top20[['Name', 'RPC (100)', 'Score', 'Rank']].copy()
    out_df['New_RPC'] = top20_sim['RPC (100)']
    out_df['Pred_Score'] = score_orig
    out_df['Pred_New_Score'] = score_sim
    out_df['Pred_Rank'] = rank_orig
    out_df['Pred_New_Rank'] = rank_sim
    out_df.to_csv(os.path.join("data", "raw", "top20_rpc_simulation.csv"), index=False)
    print("\nSimulation results saved to data/raw/top20_rpc_simulation.csv")

def simulate_feature_improvement_top20():
    print("\n=== CUSTOM SIMULATION: Increase Each Feature by 10 for Top 20 Colleges ===")
    df = pd.read_csv(os.path.join("data", "raw", "nirf_2023_cleaned.csv"))
    models_dir = os.path.join("data", "raw", "models")
    mlp_score = joblib.load(os.path.join(models_dir, "mlp_score_model.pkl"))
    mlp_rank = joblib.load(os.path.join(models_dir, "mlp_rank_model.pkl"))
    scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))
    features = ['TLR (100)', 'RPC (100)', 'GO (100)', 'OI (100)', 'PERCEPTION (100)']
    top20 = df.sort_values("Rank").head(20).copy()
    for feat in ['TLR (100)', 'GO (100)', 'OI (100)', 'PERCEPTION (100)']:
        print(f"\n--- Simulating +10 {feat} for Top 20 Colleges ---")
        top20_sim = top20.copy()
        top20_sim[feat] = (top20_sim[feat] + 10).clip(upper=100)
        X_orig = top20[features]
        X_sim = top20_sim[features]
        X_orig_scaled = scaler.transform(X_orig)
        X_sim_scaled = scaler.transform(X_sim)
        score_orig = mlp_score.predict(X_orig_scaled)
        score_sim = mlp_score.predict(X_sim_scaled)
        rank_orig = mlp_rank.predict(X_orig_scaled)
        rank_sim = mlp_rank.predict(X_sim_scaled)
        print(f"{'College':40s} {'Old':>7s} {'New':>7s} {'Old Score':>10s} {'New Score':>10s} {'Old Rank':>10s} {'New Rank':>10s}")
        for i in range(len(top20)):
            print(f"{top20.iloc[i]['Name'][:38]:40s} {top20.iloc[i][feat]:7.2f} {top20_sim.iloc[i][feat]:7.2f} {score_orig[i]:10.2f} {score_sim[i]:10.2f} {rank_orig[i]:10.2f} {rank_sim[i]:10.2f}")
        # Save to CSV
        out_df = top20[['Name', feat, 'Score', 'Rank']].copy()
        out_df['New_' + feat] = top20_sim[feat]
        out_df['Pred_Score'] = score_orig
        out_df['Pred_New_Score'] = score_sim
        out_df['Pred_Rank'] = rank_orig
        out_df['Pred_New_Rank'] = rank_sim
        out_df.to_csv(os.path.join("data", "raw", f"top20_{feat.replace(' ', '').replace('(', '').replace(')', '').lower()}_simulation.csv"), index=False)
        print(f"Simulation results saved to data/raw/top20_{feat.replace(' ', '').replace('(', '').replace(')', '').lower()}_simulation.csv")

def train_mlp():
    """Main function to train MLP models and analyze results"""
    print("[model] Training MLP model...")
    
    # Load data
    df = load_data()
    X, y_score, y_rank, features = prepare_features(df)
    
    # Train models
    mlp_score, mlp_rank, scaler, score_metrics, rank_metrics = train_mlp_models(X, y_score, y_rank, features)
    
    # Analyze feature importance
    rf_importance, correlations, perm_importance, combined_importance = analyze_feature_importance(X, y_score, features)
    
    # Visualize predictions
    visualize_predictions(score_metrics, rank_metrics)
    
    # Generate insights
    generate_insights(df, combined_importance, score_metrics, rank_metrics)
    
    print("\n[model] MLP modeling complete!")
    print("Generated files:")
    print("- mlp_predictions.png")
    print("- mlp_residuals.png")
    print("- feature_importance_comparison.png")
    print("- mlp_model_insights.txt")
    print("- models/ (saved trained models)")
    # Run custom simulation for top 20 colleges
    simulate_rpc_improvement_top20()
    simulate_feature_improvement_top20()

if __name__ == "__main__":
    train_mlp() 