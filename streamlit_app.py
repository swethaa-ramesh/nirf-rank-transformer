import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
import joblib
from datetime import datetime

# Page config
st.set_page_config(
    page_title="NIRF Rank Transformer - Interactive Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        text-align: center;
    }
    .stExpander {
        background: #f8f9fa;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load data with caching for better performance"""
    try:
        # For deployment, we'll include sample data if files don't exist
        if os.path.exists("data/raw/nirf_2023_cleaned.csv"):
            df = pd.read_csv("data/raw/nirf_2023_cleaned.csv")
        else:
            # Create sample data for demonstration
            st.warning("⚠️ Using sample data for demonstration. For full analysis, please upload your NIRF data.")
            df = create_sample_data()
        
        if os.path.exists("data/raw/college_recommendations.csv"):
            recs_df = pd.read_csv("data/raw/college_recommendations.csv")
        else:
            recs_df = create_sample_recommendations(df)
        
        # Load models if available
        models_dir = "data/raw/models"
        mlp_score = mlp_rank = scaler = None
        if os.path.exists(models_dir):
            try:
                mlp_score = joblib.load(os.path.join(models_dir, "mlp_score_model.pkl"))
                mlp_rank = joblib.load(os.path.join(models_dir, "mlp_rank_model.pkl"))
                scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))
            except:
                pass
        
        return df, recs_df, mlp_score, mlp_rank, scaler
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None, None

def create_sample_data():
    """Create sample NIRF data for demonstration"""
    np.random.seed(42)
    n_colleges = 50
    
    data = {
        'Name': [f"Sample College {i+1}" for i in range(n_colleges)],
        'State': np.random.choice(['Delhi', 'Maharashtra', 'Karnataka', 'Tamil Nadu', 'Uttar Pradesh'], n_colleges),
        'Score': np.random.normal(60, 15, n_colleges).clip(45, 85),
        'Rank': range(1, n_colleges + 1),
        'TLR (100)': np.random.normal(65, 10, n_colleges).clip(50, 85),
        'RPC (100)': np.random.normal(45, 15, n_colleges).clip(20, 80),
        'GO (100)': np.random.normal(75, 10, n_colleges).clip(60, 95),
        'OI (100)': np.random.normal(65, 8, n_colleges).clip(55, 80),
        'PERCEPTION (100)': np.random.normal(30, 20, n_colleges).clip(0, 80)
    }
    
    df = pd.DataFrame(data)
    df = df.sort_values('Score', ascending=False).reset_index(drop=True)
    df['Rank'] = range(1, len(df) + 1)
    return df

def create_sample_recommendations(df):
    """Create sample recommendations"""
    recs = []
    for idx, row in df.iterrows():
        features = ['TLR (100)', 'RPC (100)', 'GO (100)', 'OI (100)', 'PERCEPTION (100)']
        weakest = np.random.choice(features, 2, replace=False)
        
        recs.append({
            'Name': row['Name'],
            'Cluster': np.random.choice([0, 1]),
            'Current_Score': row['Score'],
            'Current_Rank': row['Rank'],
            'Weak_Feature_1': weakest[0],
            'Weak_Feature_2': weakest[1],
            'Rec_Message': f"Focus on {weakest[0]} and {weakest[1]} to improve ranking.",
            'Est_Score_Gain': np.random.uniform(2, 8),
            'Est_Rank_Improvement': np.random.uniform(1, 5)
        })
    
    return pd.DataFrame(recs)

def show_overview(df, recs_df):
    st.markdown('<div class="main-header"><h1>🎓 NIRF Rank Transformer</h1><p>Interactive Analysis Dashboard</p></div>', unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Colleges", len(df))
    with col2:
        st.metric("Average Score", f"{df['Score'].mean():.1f}")
    with col3:
        st.metric("Highest Score", f"{df['Score'].max():.1f}")
    with col4:
        st.metric("Recommendations", len(recs_df))
    
    # Top performers
    st.subheader("🏆 Top 10 Performing Colleges")
    top_10 = df.head(10)[['Rank', 'Name', 'Score', 'State']]
    st.dataframe(top_10, use_container_width=True)
    
    # Score distribution
    st.subheader("📊 Score Distribution")
    fig = px.histogram(df, x='Score', nbins=20, 
                      title="Distribution of NIRF Scores",
                      labels={'Score': 'NIRF Score', 'count': 'Number of Colleges'})
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # State performance
    st.subheader("🗺️ State-wise Performance")
    state_avg = df.groupby('State')['Score'].agg(['mean', 'count']).reset_index()
    state_avg = state_avg.sort_values('mean', ascending=False)
    
    fig = px.bar(state_avg.head(10), x='State', y='mean',
                 title="Top 10 States by Average Score",
                 labels={'mean': 'Average Score', 'State': 'State'})
    st.plotly_chart(fig, use_container_width=True)

def show_data_explorer(df):
    st.title("📊 Data Explorer")
    
    # Filters
    st.subheader("🔍 Filters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        states = ['All'] + sorted(df['State'].unique().tolist())
        selected_state = st.selectbox("Select State:", states)
    
    with col2:
        min_score, max_score = st.slider(
            "Score Range:",
            min_value=float(df['Score'].min()),
            max_value=float(df['Score'].max()),
            value=(float(df['Score'].min()), float(df['Score'].max()))
        )
    
    with col3:
        search_term = st.text_input("Search College Name:", "")
    
    # Apply filters
    filtered_df = df.copy()
    if selected_state != 'All':
        filtered_df = filtered_df[filtered_df['State'] == selected_state]
    
    filtered_df = filtered_df[
        (filtered_df['Score'] >= min_score) & 
        (filtered_df['Score'] <= max_score)
    ]
    
    if search_term:
        filtered_df = filtered_df[
            filtered_df['Name'].str.contains(search_term, case=False)
        ]
    
    st.subheader(f"📋 Results ({len(filtered_df)} colleges found)")
    st.dataframe(filtered_df[['Rank', 'Name', 'Score', 'State', 'TLR (100)', 'RPC (100)', 'GO (100)', 'OI (100)', 'PERCEPTION (100)']], 
                use_container_width=True)
    
    # Interactive scatter plot
    st.subheader("📈 Interactive Scatter Plot")
    x_axis = st.selectbox("X-axis:", ['Score', 'TLR (100)', 'RPC (100)', 'GO (100)', 'OI (100)', 'PERCEPTION (100)'])
    y_axis = st.selectbox("Y-axis:", ['RPC (100)', 'Score', 'TLR (100)', 'GO (100)', 'OI (100)', 'PERCEPTION (100)'])
    
    fig = px.scatter(filtered_df, x=x_axis, y=y_axis, 
                     hover_data=['Name', 'Rank', 'State'],
                     title=f"{x_axis} vs {y_axis}",
                     color='Score',
                     color_continuous_scale='viridis')
    st.plotly_chart(fig, use_container_width=True)

def show_recommendations(df, recs_df):
    st.title("🎯 Strategic Recommendations")
    
    # Filter recommendations
    st.subheader("🔍 Filter Recommendations")
    col1, col2 = st.columns(2)
    
    with col1:
        min_gain = st.slider("Minimum Score Gain:", 
                           float(recs_df['Est_Score_Gain'].min()), 
                           float(recs_df['Est_Score_Gain'].max()),
                           value=float(recs_df['Est_Score_Gain'].min()))
    
    with col2:
        search_college = st.text_input("Search College:", "")
    
    # Apply filters
    filtered_recs = recs_df[recs_df['Est_Score_Gain'] >= min_gain]
    if search_college:
        filtered_recs = filtered_recs[
            filtered_recs['Name'].str.contains(search_college, case=False)
        ]
    
    st.subheader(f"📋 Top Recommendations ({len(filtered_recs)} colleges)")
    
    # Sort by score gain
    filtered_recs = filtered_recs.sort_values('Est_Score_Gain', ascending=False)
    
    for idx, row in filtered_recs.head(20).iterrows():
        with st.expander(f"{row['Name']} (Rank {row['Current_Rank']:.0f})"):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Current Score", f"{row['Current_Score']:.2f}")
                st.metric("Est. Score Gain", f"+{row['Est_Score_Gain']:.2f}")
            with col2:
                st.metric("Current Rank", f"{row['Current_Rank']:.0f}")
                st.metric("Est. Rank Improvement", f"+{row['Est_Rank_Improvement']:.1f}")
            
            st.info(f"**Focus Areas:** {row['Weak_Feature_1']}, {row['Weak_Feature_2']}")
            st.write(f"**Recommendation:** {row['Rec_Message']}")
    
    # Recommendations by feature
    st.subheader("📊 Recommendations by Focus Area")
    feature_counts = pd.concat([
        filtered_recs['Weak_Feature_1'].value_counts(),
        filtered_recs['Weak_Feature_2'].value_counts()
    ]).groupby(level=0).sum()
    
    fig = px.bar(x=feature_counts.index, y=feature_counts.values,
                 title="Most Common Focus Areas in Recommendations",
                 labels={'x': 'Feature', 'y': 'Number of Recommendations'})
    st.plotly_chart(fig, use_container_width=True)

def show_simulations(df, mlp_score, mlp_rank, scaler):
    st.title("🔮 Improvement Simulations")
    
    if mlp_score is None or mlp_rank is None or scaler is None:
        st.info("💡 Simulation models not available. This feature requires trained ML models.")
        st.subheader("🎯 Manual Simulation")
        
        # Select college
        college_name = st.selectbox("Select a College:", df['Name'].tolist())
        selected_college = df[df['Name'] == college_name].iloc[0]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Score", f"{selected_college['Score']:.2f}")
        with col2:
            st.metric("Current Rank", f"{selected_college['Rank']:.0f}")
        with col3:
            st.metric("State", selected_college['State'])
        
        # Show current features
        st.subheader("📊 Current Feature Values")
        features = ['TLR (100)', 'RPC (100)', 'GO (100)', 'OI (100)', 'PERCEPTION (100)']
        
        current_values = [selected_college[feature] for feature in features]
        fig = go.Figure(data=[
            go.Bar(name='Current', x=features, y=current_values, marker_color='lightblue')
        ])
        fig.update_layout(title="Current Feature Values", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("💡 To enable AI-powered simulations, please run the full analysis pipeline locally.")
        return
    
    st.subheader("🎯 Simulate College Improvements")
    
    # Select college
    college_name = st.selectbox("Select a College:", df['Name'].tolist())
    selected_college = df[df['Name'] == college_name].iloc[0]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Score", f"{selected_college['Score']:.2f}")
    with col2:
        st.metric("Current Rank", f"{selected_college['Rank']:.0f}")
    with col3:
        st.metric("State", selected_college['State'])

def show_analytics(df):
    st.title("📈 Advanced Analytics")
    
    # Feature correlations
    st.subheader("🔗 Feature Correlations")
    features = ['TLR (100)', 'RPC (100)', 'GO (100)', 'OI (100)', 'PERCEPTION (100)', 'Score']
    corr_matrix = df[features].corr()
    
    fig = px.imshow(corr_matrix, 
                    title="Feature Correlation Matrix",
                    color_continuous_scale='RdBu',
                    aspect="auto")
    st.plotly_chart(fig, use_container_width=True)
    
    # Score ranges analysis
    st.subheader("📊 Score Range Analysis")
    df['Score_Range'] = pd.cut(df['Score'], 
                              bins=[0, 50, 60, 70, 80, 100], 
                              labels=['<50', '50-60', '60-70', '70-80', '80+'])
    
    range_counts = df['Score_Range'].value_counts().sort_index()
    fig = px.pie(values=range_counts.values, names=range_counts.index,
                 title="Distribution by Score Range")
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature distributions
    st.subheader("📈 Feature Distributions")
    selected_feature = st.selectbox("Select Feature:", features[:-1])
    
    fig = px.histogram(df, x=selected_feature, nbins=20,
                      title=f"Distribution of {selected_feature}")
    st.plotly_chart(fig, use_container_width=True)

def show_reports(df, recs_df):
    st.title("📋 Reports & Downloads")
    
    st.subheader("📊 Summary Statistics")
    
    # Overall statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Score Statistics")
        score_stats = df['Score'].describe()
        st.dataframe(score_stats)
    
    with col2:
        st.subheader("🏆 Rank Statistics")
        rank_stats = df['Rank'].describe()
        st.dataframe(rank_stats)
    
    # State-wise report
    st.subheader("🗺️ State-wise Report")
    state_report = df.groupby('State').agg({
        'Score': ['mean', 'std', 'count'],
        'Rank': ['mean', 'min', 'max']
    }).round(2)
    st.dataframe(state_report)
    
    # Download options
    st.subheader("💾 Download Reports")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Data (CSV)",
            data=csv,
            file_name="nirf_data.csv",
            mime="text/csv"
        )
    
    with col2:
        csv_recs = recs_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Recommendations (CSV)",
            data=csv_recs,
            file_name="recommendations.csv",
            mime="text/csv"
        )
    
    with col3:
        # Generate summary report
        summary_report = f"""
NIRF Rank Transformer - Analysis Report
Generated on: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}

SUMMARY STATISTICS:
- Total Colleges Analyzed: {len(df)}
- Average Score: {df['Score'].mean():.2f}
- Highest Score: {df['Score'].max():.2f}
- Lowest Score: {df['Score'].min():.2f}

TOP 5 COLLEGES:
{df.head(5)[['Rank', 'Name', 'Score']].to_string(index=False)}

RECOMMENDATIONS GENERATED: {len(recs_df)}
        """
        st.download_button(
            label="📥 Download Summary Report (TXT)",
            data=summary_report,
            file_name="analysis_summary.txt",
            mime="text/plain"
        )

def main():
    # Load data
    df, recs_df, mlp_score, mlp_rank, scaler = load_data()
    
    if df is None:
        st.error("Failed to load data. Please check your data files.")
        return
    
    # Sidebar navigation
    st.sidebar.title("🎓 NIRF Rank Transformer")
    st.sidebar.markdown("### Interactive Analysis Dashboard")
    
    # Navigation
    page = st.sidebar.selectbox(
        "Choose a Section:",
        ["🏠 Overview", "📊 Data Explorer", "🎯 Recommendations", "🔮 Simulations", "📈 Analytics", "📋 Reports"]
    )
    
    # Display selected page
    if page == "🏠 Overview":
        show_overview(df, recs_df)
    elif page == "📊 Data Explorer":
        show_data_explorer(df)
    elif page == "🎯 Recommendations":
        show_recommendations(df, recs_df)
    elif page == "🔮 Simulations":
        show_simulations(df, mlp_score, mlp_rank, scaler)
    elif page == "📈 Analytics":
        show_analytics(df)
    elif page == "📋 Reports":
        show_reports(df, recs_df)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Powered by Streamlit**")
    st.sidebar.markdown("🎓 NIRF Rank Transformer")

if __name__ == "__main__":
    main() 