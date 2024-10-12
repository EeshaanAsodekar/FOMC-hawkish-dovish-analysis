# src/data_scraper.py
import os
from FedTools import FederalReserveMins, MonetaryPolicyCommittee
import pandas as pd
import urllib.error

# Create directories to store raw and processed data
def create_directories():
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)

# Download FOMC Meeting Minutes using FedTools
def download_fomc_minutes():
    print("Downloading FOMC Meeting Minutes...")
    minutes = FederalReserveMins().find_minutes()
    minutes.to_csv('data/raw/FOMC_meeting_minutes.csv')
    print(f"Downloaded {len(minutes)} meeting minutes.")

# Download FOMC Statements using FedTools
def download_fomc_statements():
    print("Downloading FOMC Statements...")
    try:
        statements = MonetaryPolicyCommittee().find_statements()
        statements.to_csv('data/raw/FOMC_statements.csv')
        print(f"Downloaded {len(statements)} FOMC statements.")
    except urllib.error.HTTPError as e:
        print(f"Error retrieving FOMC statements: {e}")

# Execution for downloading all data
if __name__ == "__main__":
    create_directories()
    download_fomc_minutes()
    download_fomc_statements()
    print("Data scraping complete. Files saved in data/raw/")
