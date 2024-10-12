import pandas as pd

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

import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# List of market variables you want to analyze
market_vars = ['GT10_pct_change', 'GT2_pct_change', '2s10s_Spread_pct_change',
               'Gold_Prices_pct_change', 'VIX_pct_change', 'SP500_pct_change']

def run_regression_and_plot(hawkish_df, market_df, market_var):
    """
    Perform regression analysis and plot results.
    
    Parameters:
    -----------
    hawkish_df: pd.DataFrame
        DataFrame containing hawkishness scores and their changes
    market_df: pd.DataFrame
        DataFrame containing market instrument pct/abs changes
    market_var: str
        The market variable (pct change) to perform the regression on
    """

    # Ensure 'Date' column in both dataframes is of the same type (datetime64[ns])
    hawkish_df['Date'] = pd.to_datetime(hawkish_df['Date'], errors='coerce')
    market_df['Date'] = pd.to_datetime(market_df['Date'], errors='coerce')

    # Check for any NaT values (missing dates)
    if hawkish_df['Date'].isna().sum() > 0:
        print(f"Warning: {hawkish_df['Date'].isna().sum()} rows with invalid 'Date' in hawkish_df")

    # Merge the hawkishness dataframe and market data on 'Date'
    merged_df = pd.merge(hawkish_df, market_df, on='Date', how='inner')

    # Drop rows with missing data in the relevant columns
    merged_df = merged_df.dropna(subset=[market_var, 'pct_change_hawkish'])

    # Check if merged_df is empty
    if merged_df.empty:
        print(f"No valid data for {market_var} after merging hawkish_df and market_df")
        return

    # Define the independent (X) and dependent (Y) variables
    X = merged_df['pct_change_hawkish']  # Or use 'Weighted_Hawkish_Sum' for the actual score
    Y = merged_df[market_var]

    # Check if X or Y is empty
    if X.empty or Y.empty:
        print(f"Error: No valid data for regression on {market_var}.")
        return

    # Add a constant term for the regression intercept
    X_with_const = sm.add_constant(X)

    # Perform regression using OLS (Ordinary Least Squares)
    model = sm.OLS(Y, X_with_const)
    results = model.fit()

    # Print regression summary
    print(f"Regression analysis for {market_var}:")
    print(results.summary())

    # Plot 1: Scatter plot with regression line
    plt.figure(figsize=(10, 6))
    plt.scatter(X, Y, label='Data points')
    plt.plot(X, results.predict(X_with_const), color='red', label='Fitted line')
    plt.title(f"{market_var} vs pct_change_hawkish")
    plt.xlabel('pct_change_hawkish')
    plt.ylabel(market_var)
    plt.legend()
    plt.show()

    # Plot 2: Residual plot to examine goodness of fit
    plt.figure(figsize=(10, 6))
    plt.scatter(X, results.resid)
    plt.axhline(0, color='red', linestyle='--')
    plt.title(f"Residual Plot for {market_var}")
    plt.xlabel('pct_change_hawkish')
    plt.ylabel('Residuals')
    plt.show()

import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import re

def extract_date_from_filename(filename):
    """
    Extracts the date from the 'Filename' string.

    The filename is expected to have a date in the form of 'YYYYMMDD'.
    Example: 'FOMCpresconf20120425.txt' -> '20120425' -> datetime(2012, 04, 25)
    """
    # Extract the date portion (8 consecutive digits) from the string
    date_match = re.search(r'(\d{8})', filename)
    if date_match:
        return pd.to_datetime(date_match.group(1), format='%Y%m%d', errors='coerce')
    return pd.NaT

def perform_market_analysis() -> None:
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
        for market_var in market_vars:
            run_regression_and_plot(hawkish_df, mkt_data, market_var)


    run_regression_and_plot()
if __name__ == "__main__":
    perform_market_analysis()