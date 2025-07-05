import os
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time

def extract_data():
    # Check if CSV already exists
    csv_path = os.path.join("data", "raw", "nirf_2023.csv")
    if os.path.exists(csv_path):
        print(f"[scraper] {csv_path} already exists. Skipping scraping.")
        return
    base_urls = {
        2023: "https://www.nirfindia.org/2023/EngineeringRanking.html"
    }
    os.makedirs("data/raw", exist_ok=True)
    options = webdriver.ChromeOptions()
    # options.add_argument('--headless')  # Removed for visible browser
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    for year, url in base_urls.items():
        print(f"[scraper] Downloading NIRF Engineering Rankings for {year}...")
        driver.get(url)
        try:
            # Wait for the table to load
            WebDriverWait(driver, 40).until(
                EC.presence_of_element_located((By.ID, "tbl_overall"))
            )
            html = driver.page_source
            # Save the HTML for parsing
            html_path = os.path.join("data", "raw", f"nirf_{year}_table.html")
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html)
            print(f"[scraper] Saved HTML to {html_path}")
        except Exception as e:
            print(f"Error scraping {year}: {e}")
    driver.quit() 