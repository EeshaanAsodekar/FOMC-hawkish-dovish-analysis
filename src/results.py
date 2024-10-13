import pandas as pd
import os
import textwrap

def load_market_data(raw_mkt_data_file_path: str, processed_mkt_data_path: str) -> pd.DataFrame:
    """
    Loads market data from an Excel workbook, processes price columns, calculates percentage 
    and absolute changes, and outputs the result to a CSV file.

    Parameters:
    -----------
    raw_mkt_data_file_path : str
        The file path to the raw Excel workbook containing market data.
    processed_mkt_data_path : str
        The directory where the processed market data (with percentage and absolute changes) will be saved.

    Returns:
    --------
    pd.DataFrame
        A DataFrame with calculated percentage and absolute changes for each instrument, 
        aligned by date and with missing values forward-filled.
    """
    # Get sheet names from the Excel file
    excel_sheets = pd.ExcelFile(raw_mkt_data_file_path).sheet_names

    # Load each sheet into a DataFrame and store them in a dictionary
    market_data = {sheet_name: pd.read_excel(raw_mkt_data_file_path, sheet_name=sheet_name) for sheet_name in excel_sheets}

    # # Display the head and tail of each sheet for initial inspection
    # for key in market_data.keys():
    #     print(key)
    #     print(market_data[key].head())
    #     print(market_data[key].tail())
    #     print("********************")

    # Initialize list to store DataFrames for each instrument
    instruments = []

    # Extract relevant price columns from each sheet
    for key, df in market_data.items():
        # Use PX_MID for GT10 and GT2, PX_LAST for all other instruments
        price_column = 'PX_MID' if key in ["GT10", "GT2"] else 'PX_LAST'
        
        # Create DataFrame with Date and corresponding price column renamed to instrument name
        df_instrument = df[['Date', price_column]].rename(columns={price_column: key})
        
        # Add DataFrame to list
        instruments.append(df_instrument)

    # Merge all DataFrames on 'Date'
    instrument_level = instruments[0]
    for df in instruments[1:]:
        instrument_level = pd.merge(instrument_level, df, on='Date', how='outer')

    # # Display the merged DataFrame's head, tail, and missing values count
    # print(instrument_level.head())
    # print(instrument_level.tail())
    # print(instrument_level.isna().sum().sum())

    # Sort DataFrame by Date (ascending)
    instrument_level = instrument_level.sort_values(by='Date')

    # Forward fill missing values
    instrument_level = instrument_level.fillna(method='ffill')

    # Create dictionaries for percentage and absolute change column names
    pct_change_columns = {col: f'{col}_pct_change' for col in instrument_level.columns if col != 'Date'}
    abs_change_columns = {col: f'{col}_abs_change' for col in instrument_level.columns if col != 'Date'}

    # Copy DataFrame for modification
    market_moves = instrument_level.copy()

    # Calculate percentage and absolute changes for each column
    for col in instrument_level.columns:
        if col != 'Date':
            market_moves[pct_change_columns[col]] = instrument_level[col].pct_change()  # Percentage change
            market_moves[abs_change_columns[col]] = instrument_level[col].diff()  # Absolute change

    # Keep only Date, percentage change, and absolute change columns
    market_moves = market_moves[['Date'] + list(pct_change_columns.values()) + list(abs_change_columns.values())]

    # Drop rows with NaN values (typically the first row after pct_change calculation)
    market_moves = market_moves.dropna()

    # Display the final DataFrame for verification
    print("---------------------------------------------")
    print("Processed Market Data:")
    print(market_moves.head())
    print("---------------------------------------------")

    # Save the final DataFrame to a CSV file
    market_moves.to_csv(processed_mkt_data_path + "/mkt_data_pct_abs_change.csv", index=False)

    return market_moves

# List of market variables you want to analyze
market_vars = ['GT10_pct_change', 'GT2_pct_change', '2s10s_Spread_pct_change',
               'Gold_Prices_pct_change', 'VIX_pct_change', 'SP500_pct_change']

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def run_regression_and_plot_quintiles(hawkish_df, market_df, market_var, hawkish_change_col, predictor_var:str, fed_doc:str, window=5, num_quintiles=5):
    """
    Perform regression analysis and plot quintile-based results for median 5-day cumulative market changes.

    Parameters:
    -----------
    hawkish_df: pd.DataFrame
        DataFrame containing hawkishness scores and their changes.
    market_df: pd.DataFrame
        DataFrame containing market instrument pct/abs changes.
    market_var: str
        The market variable (pct change) to analyze.
    hawkish_change_col: str
        The column name in hawkish_df representing the hawkishness change (e.g., 'abs_change_hawkish' or 'pct_change_hawkish').
    window: int, optional (default=5)
        The number of available market days to calculate the cumulative market change.
    num_quintiles: int, optional (default=5)
        The number of quintiles to divide the hawkishness scores into.
    """

    # Ensure both dataframes have 'Date' column of type datetime
    hawkish_df['Date'] = pd.to_datetime(hawkish_df['Date'], errors='coerce')
    market_df['Date'] = pd.to_datetime(market_df['Date'], errors='coerce')

    # Filter hawkish_df to only include dates on or after January 1, 2012
    hawkish_df = hawkish_df[hawkish_df['Date'] >= '2012-01-01']

    # Perform a full outer join on the 'Date' column to keep all market and hawkishness data
    merged_df = pd.merge(market_df, hawkish_df[['Date', hawkish_change_col]], on='Date', how='outer')

    # Sort by Date to ensure proper sequential handling
    merged_df = merged_df.sort_values('Date')

    # Initialize a list to store cumulative changes
    cumulative_changes = []

    # Loop through the dataframe to calculate cumulative change in the market variable over the next 'window' available market days
    for i in range(len(merged_df)):
        if pd.notna(merged_df.iloc[i][hawkish_change_col]):  # Check if this row contains a hawkish score change
            current_date = merged_df.iloc[i]['Date']

            # Get the next 'window' available market days including the current date
            future_market_data = merged_df[(merged_df['Date'] >= current_date) & (pd.notna(merged_df[market_var]))]

            if len(future_market_data) >= window:
                # Sum the market variable for the next 'window' available market days
                cumulative_change = future_market_data[market_var].iloc[:window].sum()
            else:
                cumulative_change = None  # If not enough data points available for the window

            cumulative_changes.append(cumulative_change)
        else:
            cumulative_changes.append(None)  # No hawkish score change on this row

    # Add the cumulative changes as a new column
    merged_df['cumulative_change'] = cumulative_changes

    # Drop rows where cumulative changes or hawkish_change_col are missing
    merged_df = merged_df.dropna(subset=['cumulative_change', hawkish_change_col])

    # Convert hawkish_change_col to numeric to avoid any issues with mixed types
    merged_df[hawkish_change_col] = pd.to_numeric(merged_df[hawkish_change_col], errors='coerce')

    # Drop rows where hawkish_change_col is still NaN (if any exist)
    merged_df = merged_df.dropna(subset=[hawkish_change_col])

    # Divide the hawkishness scores into quintiles
    try:
        merged_df['quintile'] = pd.qcut(merged_df[hawkish_change_col], num_quintiles, labels=False)
    except ValueError as e:
        print(f"Error: {e}")
        print("Skipping this variable due to insufficient unique values for quintiles.")
        return

    # Calculate the median cumulative change for each quintile
    quintile_median = merged_df.groupby('quintile')['cumulative_change'].median()

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(quintile_median.index + 1, quintile_median, marker='o', label=market_var)
    
    title = f"{fed_doc[5:]} based - Median {window}-Day Cumulative {market_var} Across {predictor_var} Quintiles"
    wrapped_title = "\n".join(textwrap.wrap(title, width=100))
    plt.title(wrapped_title)

    plt.xlabel(f"{predictor_var} Quintile")
    plt.ylabel(f"Median {window}-Day Cumulative {market_var}")
    plt.xticks(np.arange(1, num_quintiles + 1))
    plt.legend()
    # plt.show()

    # Saving plot
    output_dir = f"data/results-vizl/{predictor_var}/"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}{fed_doc} based Median {window}-Day Cumulative {market_var} Across {predictor_var} Quintiles.png")
    plt.close()

    return quintile_median

import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import re

def extract_date_from_filename(filename):
    """
    Extracts the date from the 'Filename' string.
    
    The filename can have a date in the form of 'YYYYMMDD' or 'YYYY-MM-DD'.
    Example:
    - 'FOMCpresconf20120425.txt' -> '20120425' -> datetime(2012, 04, 25)
    - '1995-02-01_Minutes.txt' -> '1995-02-01' -> datetime(1995, 02, 01)
    """
    # Try to match 'YYYYMMDD' format
    date_match = re.search(r'(\d{8})', filename)
    if date_match:
        return pd.to_datetime(date_match.group(1), format='%Y%m%d', errors='coerce')
    else:
        # Try to match 'YYYY-MM-DD' format
        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
        if date_match:
            return pd.to_datetime(date_match.group(1), format='%Y-%m-%d', errors='coerce')
    return pd.NaT


def perform_market_analysis() -> None:
    '''
    function to orchrestrate the returns computation and do analysis 
    (quntile graphs) of the hawkishness score 1 change v/s the market instruements change
    '''
    # Process the raw market data to get pct and absolute changes
    mkt_data = load_market_data('data/raw/FOMC_Data_2011_2024.xlsx', 'data/processed')

    # Dictionary to hold the Fed communication hawkishness results
    dict_hawkish_scored = dict()

    # Load data and rename columns appropriately
    dict_hawkish_scored['dict-hawkish-scored_Fed-chair-press-conf'] = pd.read_csv(r'data/results/dict-hawkish-scored_Fed-chair-press-conf.csv').rename(columns={'Unnamed: 0': 'Filename'})
    dict_hawkish_scored['dict-hawkish-scored_Fed-speeches'] = pd.read_csv(r'data/results/dict-hawkish-scored_Fed-speeches.csv').rename(columns={'Unnamed: 0': 'Filename'})
    dict_hawkish_scored['dict-hawkish-scored_FOMC-meeting-minutes'] = pd.read_csv(r'data/results/dict-hawkish-scored_FOMC-meeting-minutes.csv').rename(columns={'Unnamed: 0': 'Filename'})
    dict_hawkish_scored['dict-hawkish-scored_FOMC-statements'] = pd.read_csv(r'data/results/dict-hawkish-scored_FOMC-statements.csv').rename(columns={'Unnamed: 0': 'Filename'})

    # Loop through the dictionaries and process each dataframe
    for key in dict_hawkish_scored.keys():
        print(f"Processing: {key}")

        # Extract date from the filename and add a new 'Date' column
        dict_hawkish_scored[key]['Date'] = dict_hawkish_scored[key]['Filename'].apply(extract_date_from_filename)

        # Calculate absolute and percentage change of the hawkish score
        dict_hawkish_scored[key]['abs_change_hawkish'] = dict_hawkish_scored[key]['Weighted_Hawkish_Sum'].diff()
        dict_hawkish_scored[key]['pct_change_hawkish'] = dict_hawkish_scored[key]['Weighted_Hawkish_Sum'].pct_change()

        # Replace inf values in pct_change_hawkish with NaN
        dict_hawkish_scored[key]['pct_change_hawkish'].replace([float('inf'), -float('inf')], pd.NA, inplace=True)

        # Optionally drop rows with NaN (including the first row after pct_change)
        dict_hawkish_scored[key] = dict_hawkish_scored[key].dropna()

        # Display the modified dataframe (for verification)
        print(dict_hawkish_scored[key].head())
        print(dict_hawkish_scored[key].shape)

    # Run regression analysis and plot for each hawkish dataframe and market variable
    for hawkish_key, hawkish_df in dict_hawkish_scored.items():
        for hawkish_change_col in ['pct_change_hawkish']:
            for market_var in market_vars:
                print(f">>>>> Plotting for: {hawkish_key} using {hawkish_change_col}")
                run_regression_and_plot_quintiles(hawkish_df, mkt_data, market_var, hawkish_change_col, "Hawkishness-score-1", hawkish_key)


def perform_market_analysis_hawk2() -> None:
    '''
    function to orchrestrate the returns computation and do analysis 
    (quntile graphs) of the hawkishness score 2 change v/s the market instruements change
    '''
    # Process the raw market data to get pct and absolute changes
    mkt_data = load_market_data('data/raw/FOMC_Data_2011_2024.xlsx', 'data/processed')

    # Dictionary to hold the Fed communication hawkishness results
    dict_hawkish_scored = dict()

    # Load data and rename columns appropriately
    dict_hawkish_scored['dict-hawkish-scored_Fed-chair-press-conf'] = pd.read_csv(r'data/results/dict-hawkish-scored_Fed-chair-press-conf_hdict2.csv').rename(columns={'Unnamed: 0': 'Filename'})
    dict_hawkish_scored['dict-hawkish-scored_Fed-speeches'] = pd.read_csv(r'data/results/dict-hawkish-scored_Fed-speeches_hdict2.csv').rename(columns={'Unnamed: 0': 'Filename'})
    dict_hawkish_scored['dict-hawkish-scored_FOMC-meeting-minutes'] = pd.read_csv(r'data/results/dict-hawkish-scored_FOMC-meeting-minutes_hdict2.csv').rename(columns={'Unnamed: 0': 'Filename'})
    dict_hawkish_scored['dict-hawkish-scored_FOMC-statements'] = pd.read_csv(r'data/results/dict-hawkish-scored_FOMC-statements_hdict2.csv').rename(columns={'Unnamed: 0': 'Filename'})

    # Loop through the dictionaries and process each dataframe
    for key in dict_hawkish_scored.keys():
        print(f"Processing: {key}")

        # Extract date from the filename and add a new 'Date' column
        dict_hawkish_scored[key]['Date'] = dict_hawkish_scored[key]['Filename'].apply(extract_date_from_filename)

        # Calculate absolute and percentage change of the hawkish score
        dict_hawkish_scored[key]['abs_change_hawkish'] = dict_hawkish_scored[key]['Weighted_Hawkish_Sum'].diff()
        dict_hawkish_scored[key]['pct_change_hawkish'] = dict_hawkish_scored[key]['Weighted_Hawkish_Sum'].pct_change()

        # Replace inf values in pct_change_hawkish with NaN
        dict_hawkish_scored[key]['pct_change_hawkish'].replace([float('inf'), -float('inf')], pd.NA, inplace=True)

        # Optionally drop rows with NaN (including the first row after pct_change)
        dict_hawkish_scored[key] = dict_hawkish_scored[key].dropna()

        # Display the modified dataframe (for verification)
        print(dict_hawkish_scored[key].head())
        print(dict_hawkish_scored[key].shape)

    # Run regression analysis and plot for each hawkish dataframe and market variable
    for hawkish_key, hawkish_df in dict_hawkish_scored.items():
        for hawkish_change_col in ['pct_change_hawkish']:
            for market_var in market_vars:
                print(f">>>>> Plotting for: {hawkish_key} using {hawkish_change_col}")
                run_regression_and_plot_quintiles(hawkish_df, mkt_data, market_var, hawkish_change_col, "Hawkishness-score-2", hawkish_key)


def perform_market_analysis_composite() -> None:
    '''
    function to orchrestrate the returns computation and do analysis 
    (quntile graphs) of the composite score change v/s the market instruements change
    '''
    # Process the raw market data to get pct and absolute changes
    mkt_data = load_market_data('data/raw/FOMC_Data_2011_2024.xlsx', 'data/processed')

    # Dictionary to hold the Fed communication hawkishness results
    dict_hawkish_scored = dict()

    # Load data and rename columns appropriately
    dict_hawkish_scored['composite-scored_Fed-chair-press-conf'] = pd.read_csv(r'data/results/composite-scored_Fed-chair-press-conf.csv').rename(columns={'Unnamed: 0': 'Filename'})
    dict_hawkish_scored['composite-scored_Fed-speeches'] = pd.read_csv(r'data/results/composite-scored_Fed-speeches_hdict2.csv').rename(columns={'Unnamed: 0': 'Filename'})
    dict_hawkish_scored['composite-scored_FOMC-meeting-minutes'] = pd.read_csv(r'data/results/composite-scored_FOMC-meeting-minutes_hdict2.csv').rename(columns={'Unnamed: 0': 'Filename'})
    dict_hawkish_scored['composite-scored_FOMC-statements'] = pd.read_csv(r'data/results/composite-scored_FOMC-statements_hdict2.csv').rename(columns={'Unnamed: 0': 'Filename'})

    # Loop through the dictionaries and process each dataframe
    for key in dict_hawkish_scored.keys():
        print(f"Processing: {key}")

        # Extract date from the filename and add a new 'Date' column
        dict_hawkish_scored[key]['Date'] = dict_hawkish_scored[key]['Filename'].apply(extract_date_from_filename)

        # Calculate absolute and percentage change of the hawkish score
        dict_hawkish_scored[key]['abs_change_hawkish'] = dict_hawkish_scored[key]['Composite_Score'].diff()
        dict_hawkish_scored[key]['pct_change_hawkish'] = dict_hawkish_scored[key]['Composite_Score'].pct_change()

        # Replace inf values in pct_change_hawkish with NaN
        dict_hawkish_scored[key]['pct_change_hawkish'].replace([float('inf'), -float('inf')], pd.NA, inplace=True)

        # Optionally drop rows with NaN (including the first row after pct_change)
        dict_hawkish_scored[key] = dict_hawkish_scored[key].dropna()

        # Display the modified dataframe (for verification)
        print(dict_hawkish_scored[key].head())
        print(dict_hawkish_scored[key].shape)

    # Run regression analysis and plot for each hawkish dataframe and market variable
    i = 0
    for hawkish_key, hawkish_df in dict_hawkish_scored.items():
        for hawkish_change_col in ['pct_change_hawkish']:
            for market_var in market_vars:
                print(f">>>>> Plotting for: {hawkish_key} using {hawkish_change_col}")
                run_regression_and_plot_quintiles(hawkish_df, mkt_data, market_var, hawkish_change_col,"Composite-score", hawkish_key)
                i+=1

    print("total plots>> : ", i)

if __name__ == "__main__":
    # Analysis on the original hawkish score
    perform_market_analysis()

    # Analysis on the hawkish score 2
    perform_market_analysis_hawk2()

    # Analysis on the composite score
    perform_market_analysis_composite()