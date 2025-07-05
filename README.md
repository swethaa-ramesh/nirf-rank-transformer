# NIRF Rank Transformer

## Overview

This project analyzes and transforms NIRF (National Institutional Ranking Framework) rankings for 500+ colleges using advanced data analysis and deep learning. It extracts, processes, and models ranking parameters to deliver actionable recommendations for ranking improvement.

## Objectives

- Extract comprehensive NIRF datasets via web scraping
- Analyze critical ranking factors (faculty-student ratio, placements, etc.)
- Model feature influence using MLP (Multi-Layer Perceptron)
- Cluster similar colleges for peer-based recommendations
- Deliver actionable insights to improve rankings

## Pipeline

1. **Data Extraction:** Scrape NIRF data for 500+ colleges and ranking parameters
2. **Preprocessing:** Clean and prepare the data for analysis
3. **EDA:** Explore and visualize key factors and correlations
4. **Modeling:** Build MLP model for feature importance analysis
5. **Clustering:** Group similar colleges for peer analysis
6. **Recommendations:** Generate actionable insights for ranking improvement

## EDA Insights Summary

### Dataset Overview

- **Total Colleges:** 100 (NIRF 2023 Engineering Rankings)
- **Features:** Institute ID, Name, TLR (100), RPC (100), GO (100), OI (100), PERCEPTION (100), City, State, Score, Rank

### Key Findings

#### 1. Top 10 Colleges by NIRF Score

- **IIT Madras** leads with a score of 86.69
- **IISc Bangalore** follows with 83.09
- **IIT Delhi** ranks third with 82.16
- Top colleges are dominated by IITs and IISc
- Significant score gap exists between top 3 and remaining colleges

#### 2. Feature Distributions

- **Score:** Mean ~55, most colleges cluster around the mean with few high outliers
- **TLR (Teaching, Learning & Resources):** Mean ~64, skewed towards higher values for top colleges
- **RPC (Research & Professional Practice):** Mean ~41, wide spread with top colleges scoring much higher
- **GO (Graduation Outcomes):** Mean ~74, generally high across colleges
- **OI (Outreach & Inclusivity):** Mean ~62, moderate spread
- **PERCEPTION:** Mean ~25, highly skewed with only few colleges having very high perception scores

#### 3. Feature Correlations

- **Score** is most strongly correlated with TLR, RPC, and GO
- **PERCEPTION** has weaker correlation with overall score, indicating it's a differentiator for only select colleges
- **RPC** and **GO** show positive correlation
- **OI** has moderate correlation with overall score

#### 4. Ranking Insights

- Top-ranked colleges excel in multiple parameters simultaneously
- RPC (Research) is a key differentiator for top-tier institutions
- PERCEPTION scores vary dramatically, suggesting reputation building opportunities
- TLR scores are relatively high across institutions, indicating good teaching infrastructure

### Generated Visualizations

- `top10_colleges.png` - Bar chart of top 10 colleges by score
- `dist_Score.png` - Score distribution histogram
- `dist_TLR_100.png` - TLR distribution
- `dist_RPC_100.png` - RPC distribution
- `dist_GO_100.png` - GO distribution
- `dist_OI_100.png` - OI distribution
- `dist_PERCEPTION_100.png` - PERCEPTION distribution
- `correlation_heatmap.png` - Feature correlation matrix

## Files Structure

```
NIRF/
├── data/
│   └── raw/
│       ├── nirf_2023.csv              # Raw extracted data
│       ├── nirf_2023_cleaned.csv      # Cleaned dataset
│       └── *.png                      # EDA visualizations
├── src/
│   ├── scraper.py                     # Web scraping module
│   ├── preprocess.py                  # Data cleaning and EDA
│   ├── model.py                       # MLP modeling
│   ├── cluster.py                     # Clustering analysis
│   └── recommend.py                   # Recommendation engine
├── main.py                            # Main pipeline script
├── parse_nirf_html.py                 # HTML parser for NIRF data
└── README.md                          # Project documentation
```

## Usage

1. Install dependencies: `pip install -r requirements.txt`
2. Run the complete pipeline: `python main.py`
3. View generated visualizations in `data/raw/`

## Next Steps

- Implement MLP modeling for feature importance analysis
- Perform clustering to group similar colleges
- Generate actionable recommendations for ranking improvement
