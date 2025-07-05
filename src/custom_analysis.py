import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

def load_data():
    """Load the cleaned NIRF data"""
    data_path = os.path.join("data", "raw", "nirf_2023_cleaned.csv")
    return pd.read_csv(data_path)

def state_wise_analysis(df):
    """Analyze performance by state"""
    print("\n=== STATE-WISE ANALYSIS ===")
    
    # State-wise statistics
    state_stats = df.groupby('State').agg({
        'Score': ['count', 'mean', 'std', 'min', 'max'],
        'Rank': ['mean', 'min', 'max'],
        'TLR (100)': 'mean',
        'RPC (100)': 'mean',
        'GO (100)': 'mean',
        'OI (100)': 'mean',
        'PERCEPTION (100)': 'mean'
    }).round(2)
    
    # Flatten column names
    state_stats.columns = ['_'.join(col).strip() for col in state_stats.columns]
    state_stats = state_stats.reset_index()
    
    # Top 10 states by average score
    top_states = state_stats.nlargest(10, 'Score_mean')
    
    # Plot 1: Top 10 states by average score
    plt.figure(figsize=(12, 6))
    bars = plt.barh(top_states['State'], top_states['Score_mean'], color='skyblue')
    plt.xlabel('Average Score')
    plt.title('Top 10 States by Average NIRF Score (2023)')
    plt.gca().invert_yaxis()
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                f'{width:.1f}', ha='left', va='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join("data", "raw", "top_states_by_score.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Number of colleges per state
    state_counts = df['State'].value_counts().head(15)
    plt.figure(figsize=(12, 6))
    bars = plt.bar(state_counts.index, state_counts.values, color='lightcoral')
    plt.xlabel('State')
    plt.ylabel('Number of Colleges')
    plt.title('Number of Colleges by State (Top 15)')
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{int(height)}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join("data", "raw", "colleges_per_state.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Top 5 states by average score:")
    for i, row in top_states.head().iterrows():
        print(f"{i+1}. {row['State']}: {row['Score_mean']:.2f} (avg), {row['Score_count']} colleges")
    
    return state_stats

def score_gap_analysis(df):
    """Analyze score gaps and improvement potential"""
    print("\n=== SCORE GAP ANALYSIS ===")
    
    # Calculate expected score based on feature averages
    features = ['TLR (100)', 'RPC (100)', 'GO (100)', 'OI (100)', 'PERCEPTION (100)']
    df['Expected_Score'] = df[features].mean(axis=1)
    df['Score_Gap'] = df['Score'] - df['Expected_Score']
    df['Performance_Category'] = pd.cut(df['Score_Gap'], 
                                       bins=[-float('inf'), -5, 5, float('inf')],
                                       labels=['Underperformer', 'Expected', 'Overperformer'])
    
    # Plot 1: Score vs Expected Score
    plt.figure(figsize=(10, 8))
    colors = {'Underperformer': 'red', 'Expected': 'blue', 'Overperformer': 'green'}
    
    for category in df['Performance_Category'].unique():
        subset = df[df['Performance_Category'] == category]
        plt.scatter(subset['Expected_Score'], subset['Score'], 
                   c=colors[category], label=category, alpha=0.7, s=50)
    
    # Add diagonal line (perfect prediction)
    min_val = min(df['Expected_Score'].min(), df['Score'].min())
    max_val = max(df['Expected_Score'].max(), df['Score'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect Prediction')
    
    plt.xlabel('Expected Score (Average of Features)')
    plt.ylabel('Actual Score')
    plt.title('Actual vs Expected Scores: Performance Analysis')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join("data", "raw", "score_gap_analysis.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Score gap distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df['Score_Gap'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(x=0, color='red', linestyle='--', label='Expected Performance')
    plt.xlabel('Score Gap (Actual - Expected)')
    plt.ylabel('Number of Colleges')
    plt.title('Distribution of Score Gaps')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join("data", "raw", "score_gap_distribution.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Top overperformers and underperformers
    top_overperformers = df.nlargest(10, 'Score_Gap')[['Name', 'Score', 'Expected_Score', 'Score_Gap', 'Rank']]
    top_underperformers = df.nsmallest(10, 'Score_Gap')[['Name', 'Score', 'Expected_Score', 'Score_Gap', 'Rank']]
    
    print(f"\nTop 5 Overperformers:")
    for i, row in top_overperformers.head().iterrows():
        print(f"{i+1}. {row['Name']}: Gap = {row['Score_Gap']:.2f}, Rank = {row['Rank']}")
    
    print(f"\nTop 5 Underperformers:")
    for i, row in top_underperformers.head().iterrows():
        print(f"{i+1}. {row['Name']}: Gap = {row['Score_Gap']:.2f}, Rank = {row['Rank']}")
    
    return df

def feature_importance_preview(df):
    """Preview feature importance using Random Forest"""
    print("\n=== FEATURE IMPORTANCE PREVIEW ===")
    
    # Prepare features and target
    features = ['TLR (100)', 'RPC (100)', 'GO (100)', 'OI (100)', 'PERCEPTION (100)']
    X = df[features]
    y = df['Score']
    
    # Train Random Forest for feature importance
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # Get feature importance
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    bars = plt.barh(importance_df['Feature'], importance_df['Importance'], color='lightgreen')
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance for NIRF Score Prediction (Random Forest)')
    plt.gca().invert_yaxis()
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', ha='left', va='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join("data", "raw", "feature_importance.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Correlation with Score
    correlations = df[features + ['Score']].corr()['Score'].drop('Score').sort_values(ascending=False)
    
    plt.figure(figsize=(10, 6))
    bars = plt.barh(correlations.index, correlations.values, color='lightcoral')
    plt.xlabel('Correlation with Score')
    plt.title('Feature Correlations with NIRF Score')
    plt.gca().invert_yaxis()
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', ha='left', va='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join("data", "raw", "feature_correlations.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Feature Importance (Random Forest):")
    for i, row in importance_df.iterrows():
        print(f"{i+1}. {row['Feature']}: {row['Importance']:.3f}")
    
    print(f"\nFeature Correlations with Score:")
    for feature, corr in correlations.items():
        print(f"{feature}: {corr:.3f}")
    
    return importance_df, correlations

def competitive_analysis(df):
    """Analyze colleges within similar score ranges"""
    print("\n=== COMPETITIVE ANALYSIS ===")
    
    # Create score ranges
    df['Score_Range'] = pd.cut(df['Score'], 
                              bins=[0, 50, 60, 70, 80, 100], 
                              labels=['<50', '50-60', '60-70', '70-80', '80+'])
    
    # Analyze each range
    range_analysis = df.groupby('Score_Range').agg({
        'Name': 'count',
        'Score': ['mean', 'std'],
        'TLR (100)': 'mean',
        'RPC (100)': 'mean',
        'GO (100)': 'mean',
        'OI (100)': 'mean',
        'PERCEPTION (100)': 'mean'
    }).round(2)
    
    # Flatten column names
    range_analysis.columns = ['_'.join(col).strip() for col in range_analysis.columns]
    range_analysis = range_analysis.reset_index()
    
    # Plot score range distribution
    plt.figure(figsize=(10, 6))
    range_counts = df['Score_Range'].value_counts().sort_index()
    bars = plt.bar(range_counts.index, range_counts.values, color='gold')
    plt.xlabel('Score Range')
    plt.ylabel('Number of Colleges')
    plt.title('Distribution of Colleges by Score Range')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{int(height)}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join("data", "raw", "score_range_distribution.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot feature profiles by score range
    features = ['TLR (100)', 'RPC (100)', 'GO (100)', 'OI (100)', 'PERCEPTION (100)']
    range_means = df.groupby('Score_Range')[features].mean()
    
    plt.figure(figsize=(12, 8))
    range_means.plot(kind='bar', figsize=(12, 8))
    plt.xlabel('Score Range')
    plt.ylabel('Average Score')
    plt.title('Feature Profiles by Score Range')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join("data", "raw", "feature_profiles_by_range.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Score Range Analysis:")
    for i, row in range_analysis.iterrows():
        print(f"{row['Score_Range']}: {row['Name_count']} colleges, avg score = {row['Score_mean']:.2f}")
    
    return range_analysis

def improvement_opportunities(df):
    """Identify improvement opportunities for colleges"""
    print("\n=== IMPROVEMENT OPPORTUNITIES ===")
    
    features = ['TLR (100)', 'RPC (100)', 'GO (100)', 'OI (100)', 'PERCEPTION (100)']
    
    # Find colleges with lowest scores in each feature
    opportunities = {}
    for feature in features:
        lowest_colleges = df.nsmallest(10, feature)[['Name', feature, 'Score', 'Rank']]
        opportunities[feature] = lowest_colleges
    
    # Plot improvement potential heatmap
    plt.figure(figsize=(12, 8))
    
    # Calculate improvement potential (how much below average each college is in each feature)
    improvement_data = df.copy()
    for feature in features:
        mean_score = df[feature].mean()
        improvement_data[f'{feature}_Improvement'] = mean_score - df[feature]
    
    # Select top 20 colleges by overall improvement potential
    improvement_cols = [f'{feature}_Improvement' for feature in features]
    improvement_data['Total_Improvement'] = improvement_data[improvement_cols].sum(axis=1)
    top_improvement = improvement_data.nlargest(20, 'Total_Improvement')
    
    # Create heatmap
    heatmap_data = top_improvement[improvement_cols].T
    heatmap_data.columns = top_improvement['Name'].str[:20] + '...'  # Truncate names
    
    sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='Reds', 
                cbar_kws={'label': 'Improvement Potential'})
    plt.title('Top 20 Colleges: Improvement Potential by Feature')
    plt.xlabel('College (truncated names)')
    plt.ylabel('Feature')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join("data", "raw", "improvement_opportunities_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Quick wins analysis (colleges with high overall score but low in specific features)
    high_scorers = df[df['Score'] > df['Score'].quantile(0.7)]  # Top 30%
    quick_wins = {}
    
    for feature in features:
        feature_low = high_scorers.nsmallest(5, feature)[['Name', feature, 'Score', 'Rank']]
        quick_wins[feature] = feature_low
    
    print("Top 5 Colleges with Most Improvement Potential:")
    for i, row in improvement_data.nlargest(5, 'Total_Improvement').iterrows():
        print(f"{i+1}. {row['Name']}: Total improvement potential = {row['Total_Improvement']:.1f}")
    
    print(f"\nQuick Wins (High-scoring colleges with low specific features):")
    for feature, colleges in quick_wins.items():
        print(f"\n{feature}:")
        for j, row in colleges.iterrows():
            print(f"  - {row['Name']}: {row[feature]:.1f} (Score: {row['Score']:.1f})")
    
    return opportunities, quick_wins

def run_custom_analysis():
    """Run all custom analyses"""
    print("Starting Custom Analysis...")
    
    # Load data
    df = load_data()
    
    # Run all analyses
    state_stats = state_wise_analysis(df)
    df_with_gaps = score_gap_analysis(df)
    importance_df, correlations = feature_importance_preview(df)
    range_analysis = competitive_analysis(df)
    opportunities, quick_wins = improvement_opportunities(df)
    
    # Save summary to file
    summary_path = os.path.join("data", "raw", "custom_analysis_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("NIRF 2023 CUSTOM ANALYSIS SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("1. STATE-WISE ANALYSIS\n")
        f.write("-" * 20 + "\n")
        f.write("Top 5 states by average score:\n")
        top_states = state_stats.nlargest(5, 'Score_mean')
        for i, row in top_states.iterrows():
            f.write(f"{i+1}. {row['State']}: {row['Score_mean']:.2f} (avg), {row['Score_count']} colleges\n")
        
        f.write("\n2. FEATURE IMPORTANCE\n")
        f.write("-" * 20 + "\n")
        for i, row in importance_df.iterrows():
            f.write(f"{i+1}. {row['Feature']}: {row['Importance']:.3f}\n")
        
        f.write("\n3. IMPROVEMENT OPPORTUNITIES\n")
        f.write("-" * 20 + "\n")
        f.write("Top 5 colleges with most improvement potential:\n")
        improvement_data = df_with_gaps.copy()
        features = ['TLR (100)', 'RPC (100)', 'GO (100)', 'OI (100)', 'PERCEPTION (100)']
        for feature in features:
            improvement_data[f'{feature}_Improvement'] = df_with_gaps[feature].mean() - df_with_gaps[feature]
        improvement_cols = [f'{feature}_Improvement' for feature in features]
        improvement_data['Total_Improvement'] = improvement_data[improvement_cols].sum(axis=1)
        for i, row in improvement_data.nlargest(5, 'Total_Improvement').iterrows():
            f.write(f"{i+1}. {row['Name']}: {row['Total_Improvement']:.1f}\n")
    
    print(f"\nCustom analysis complete! Summary saved to {summary_path}")
    print("Generated visualizations:")
    print("- top_states_by_score.png")
    print("- colleges_per_state.png") 
    print("- score_gap_analysis.png")
    print("- score_gap_distribution.png")
    print("- feature_importance.png")
    print("- feature_correlations.png")
    print("- score_range_distribution.png")
    print("- feature_profiles_by_range.png")
    print("- improvement_opportunities_heatmap.png")

if __name__ == "__main__":
    run_custom_analysis() 