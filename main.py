from src import scraper, preprocess, model, cluster, recommend, dashboard
from src import custom_analysis

def main():
    print("Step 1: Data Extraction")
    scraper.extract_data()

    print("Step 2: Preprocessing")
    preprocess.clean_data()

    print("Step 3: Exploratory Data Analysis (EDA)")
    preprocess.eda()

    print("Step 4: Custom Analysis")
    custom_analysis.run_custom_analysis()

    print("Step 5: Modeling (MLP)")
    model.train_mlp()

    print("Step 6: Clustering")
    cluster.run_clustering()

    print("Step 7: Recommendations")
    recommend.generate_recommendations()

    print("Step 8: Dashboard")
    dashboard.create_dashboard()

if __name__ == "__main__":
    main() 