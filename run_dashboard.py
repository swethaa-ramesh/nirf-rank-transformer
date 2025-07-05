#!/usr/bin/env python3
"""
NIRF Rank Transformer - Interactive Dashboard Launcher
"""

import subprocess
import sys
import os

def main():
    print("🎓 NIRF Rank Transformer - Interactive Dashboard")
    print("=" * 50)
    
    # Check if streamlit is installed
    try:
        import streamlit
        print("✅ Streamlit is installed")
    except ImportError:
        print("❌ Streamlit not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit", "plotly"])
        print("✅ Streamlit installed successfully")
    
    # Check if plotly is installed
    try:
        import plotly
        print("✅ Plotly is installed")
    except ImportError:
        print("❌ Plotly not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "plotly"])
        print("✅ Plotly installed successfully")
    
    # Check if data files exist
    required_files = [
        "data/raw/nirf_2023_cleaned.csv",
        "data/raw/college_recommendations.csv"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required data files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        print("\nPlease run the main pipeline first:")
        print("   python main.py")
        return
    
    print("✅ All required files found")
    
    # Launch the dashboard
    print("\n🚀 Launching Interactive Dashboard...")
    print("📱 The dashboard will open in your default web browser")
    print("🔄 To stop the dashboard, press Ctrl+C in this terminal")
    print("=" * 50)
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "src/interactive_dashboard.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")

if __name__ == "__main__":
    main() 