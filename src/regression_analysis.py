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

from scipy.stats import zscore
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import os

import statsmodels.api as sm
import matplotlib.pyplot as plt
import os

def run_regression_compute_stats(hawkish_df, market_df, market_var, hawkish_change_col, predictor_var, fed_doc, window=5):
    """
    Perform regression analysis and compute statistical metrics for the given market variable and hawkish score changes.

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
    predictor_var: str
        The name of the predictor variable (e.g., Hawkishness-score-1) for labeling.
    fed_doc: str
        Name of the Fed document for labeling the output.
    window: int, optional (default=5)
        The number of available market days to calculate the cumulative market change.
    """

    # Ensure both dataframes have 'Date' column of type datetime
    hawkish_df['Date'] = pd.to_datetime(hawkish_df['Date'], errors='coerce')
    market_df['Date'] = pd.to_datetime(market_df['Date'], errors='coerce')

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

    # Convert hawkish_change_col to numeric to avoid any issues with mixed types
    merged_df[hawkish_change_col] = pd.to_numeric(merged_df[hawkish_change_col], errors='coerce')

    # Drop rows where hawkish_change_col or cumulative_change are NaN
    merged_df = merged_df.dropna(subset=['cumulative_change', hawkish_change_col])

    # Define the independent (X) and dependent (Y) variables for the regression
    X = merged_df[hawkish_change_col]
    Y = merged_df['cumulative_change']

    # Ensure there is enough data after dropping NaNs
    if X.empty or Y.empty:
        print(f"Error: No valid data available for {fed_doc} with {market_var} after removing invalid values.")
        return


    # Add a constant term (intercept) to the model
    X_with_const = sm.add_constant(X)

    # Fit an Ordinary Least Squares (OLS) regression model
    model = sm.OLS(Y, X_with_const)
    results = model.fit()

    # Extract R-squared and other statistics
    r_squared = results.rsquared
    coeff = results.params[1]  # Coefficient for the hawkish change
    p_value = results.pvalues[1]
    const = results.params[0]
    std_err = results.bse[1]

    # Print regression summary (optional)
    print(f"Regression analysis for {market_var}:")
    print(results.summary())

    # Plot the regression line and annotate with R-squared
    plt.figure(figsize=(10, 6))
    plt.scatter(X, Y, label='Data points')
    plt.plot(X, results.predict(X_with_const), color='red', label=f'Fitted line (R² = {r_squared:.3f})')
    
    title = f"{fed_doc[5:]} based - Regression {market_var} vs {predictor_var}"
    wrapped_title = "\n".join(textwrap.wrap(title, width=100))
    plt.title(wrapped_title)
    
    plt.xlabel(predictor_var)
    plt.ylabel(f"Cumulative {market_var}")
    plt.legend()
    
    # Save the plot
    output_dir = f"data/results-regression/{predictor_var}/"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}{fed_doc} based Regression {market_var} vs {predictor_var}.png")
    plt.close()

    return {
        'R_squared': r_squared,
        'Coefficient': coeff,
        'P_value': p_value,
        'Constant': const,
        'Standard_Error': std_err
    }





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

    r_squared_populate = []
    # Run regression analysis and plot for each hawkish dataframe and market variable
    for hawkish_key, hawkish_df in dict_hawkish_scored.items():
        for hawkish_change_col in ['pct_change_hawkish']:
            for market_var in market_vars:
                print(f">>>>> Plotting for: {hawkish_key} using {hawkish_change_col}")
                res = run_regression_compute_stats(hawkish_df, mkt_data, market_var, hawkish_change_col, "Hawkishness-score-1", hawkish_key)
                print(res)
                r_squared_populate.append(res.get('R_squared'))

    for rsq in r_squared_populate:
        print(rsq)

def perform_market_analysis_dov() -> None:
    '''
    function to orchrestrate the returns computation and do analysis 
    (quntile graphs) of the dovishness score 2 change v/s the market instruements change
    '''
    # Process the raw market data to get pct and absolute changes
    mkt_data = load_market_data('data/raw/FOMC_Data_2011_2024.xlsx', 'data/processed')

    # Dictionary to hold the Fed communication hawkishness results
    dict_hawkish_scored = dict()

    # Load data and rename columns appropriately
    dict_hawkish_scored['dict-dovish-scored_Fed-chair-press-conf'] = pd.read_csv(r'data\results\dict-dovish-scored_Fed-chair-press-conf.csv').rename(columns={'Unnamed: 0': 'Filename'})
    dict_hawkish_scored['dict-dovish-scored_Fed-speeches'] = pd.read_csv(r'data\results\dict-dovish-scored_Fed-speeches_hdict2.csv').rename(columns={'Unnamed: 0': 'Filename'})
    dict_hawkish_scored['dict-dovish-scored_FOMC-meeting-minutes'] = pd.read_csv(r'data\results\dict-dovish-scored_FOMC-meeting-minutes_hdict2.csv').rename(columns={'Unnamed: 0': 'Filename'})
    dict_hawkish_scored['dict-dovish-scored_FOMC-statements'] = pd.read_csv(r'data\results\dict-dovish-scored_FOMC-statements_hdict2.csv').rename(columns={'Unnamed: 0': 'Filename'})

    # Loop through the dictionaries and process each dataframe
    for key in dict_hawkish_scored.keys():
        print(f"Processing: {key}")

        # Extract date from the filename and add a new 'Date' column
        dict_hawkish_scored[key]['Date'] = dict_hawkish_scored[key]['Filename'].apply(extract_date_from_filename)

        # Calculate absolute and percentage change of the hawkish score
        dict_hawkish_scored[key]['abs_change_hawkish'] = dict_hawkish_scored[key]['Weighted_Dovish_Sum'].diff()
        dict_hawkish_scored[key]['pct_change_hawkish'] = dict_hawkish_scored[key]['Weighted_Dovish_Sum'].pct_change()

        # Replace inf values in pct_change_hawkish with NaN
        dict_hawkish_scored[key]['pct_change_hawkish'].replace([float('inf'), -float('inf')], pd.NA, inplace=True)

        # Optionally drop rows with NaN (including the first row after pct_change)
        dict_hawkish_scored[key] = dict_hawkish_scored[key].dropna()

        # Display the modified dataframe (for verification)
        print(dict_hawkish_scored[key].head())
        print(dict_hawkish_scored[key].shape)

    i = 0
    r_squared_populate = []
    # Run regression analysis and plot for each hawkish dataframe and market variable
    for hawkish_key, hawkish_df in dict_hawkish_scored.items():
        for hawkish_change_col in ['pct_change_hawkish']:
            for market_var in market_vars:
                print(f">>>>> Plotting for: {hawkish_key} using {hawkish_change_col}")
                run_regression_compute_stats(hawkish_df, mkt_data, market_var, hawkish_change_col, "Dovish-score", hawkish_key)
                i+=1
                res = run_regression_compute_stats(hawkish_df, mkt_data, market_var, hawkish_change_col, "Hawkishness-score-1", hawkish_key)
                print(res)
                r_squared_populate.append(res.get('R_squared'))

    for rsq in r_squared_populate:
        print(rsq)
    
    print("tot dov plots: ", i)

if __name__ == "__main__":
    # Analysis on the original hawkish score
    perform_market_analysis()

    # Analysis on the dovish score
    perform_market_analysis_dov()

### previous approach
# import pandas as pd
# import statsmodels.api as sm
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler

# # Load FOMC Scores
# fomc_minutes = pd.read_csv(r'data\processed\cosine_sim_H-D-score_meeting_minutes.csv')  # FOMC Meeting Minutes

# # Load Market Data
# market_data = pd.read_csv(r'data\raw\market_data_1994_2024.csv')  # The 10Y-2Y spread, 1Y Treasury, DXY, Growth-Value Spread

# # Convert Date columns to datetime for proper merging
# fomc_minutes['Date'] = pd.to_datetime(fomc_minutes['Date'])
# market_data['Date'] = pd.to_datetime(market_data['Date']).dt.date  # Remove time part from market data

# # Convert FOMC 'Date' to date (remove timestamp if any)
# fomc_minutes['Date'] = fomc_minutes['Date'].dt.date

# # Merge FOMC scores with market data on Date
# merged_df = pd.merge(fomc_minutes, market_data, on='Date', how='left')

# # Forward and Backward fill for missing market data
# merged_df = merged_df.fillna(method='ffill').fillna(method='bfill')

# # Select relevant columns for regression
# merged_df = merged_df[['Date', 'Hawkish_Score', 'Dovish_Score', '10Y_Yield', '2Y_Yield', '10Y-2Y_Spread', '1Y_Yield']]

# # Define X (independent variables) and Y (dependent variables)
# X = merged_df[['10Y-2Y_Spread', '2Y_Yield', '1Y_Yield']]  # Market Variables (excluding DXY and Growth-Value Spread for now)
# Y_hawkish = merged_df['Hawkish_Score']  # Dependent Variable: Hawkish Score
# Y_dovish = merged_df['Dovish_Score']    # Dependent Variable: Dovish Score

# # Standardize the independent variables (X)
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Add constant for regression intercept
# X_with_const = sm.add_constant(X_scaled)

# # Perform regression for Hawkish Score
# model_hawkish = sm.OLS(Y_hawkish, X_with_const)
# results_hawkish = model_hawkish.fit()
# print(results_hawkish.summary())

# # Perform regression for Dovish Score
# model_dovish = sm.OLS(Y_dovish, X_with_const)
# results_dovish = model_dovish.fit()
# print(results_dovish.summary())

# # Print R-squared scores
# print(f"Hawkish R-squared: {results_hawkish.rsquared}")
# print(f"Dovish R-squared: {results_dovish.rsquared}")

# # Plotting results for visual analysis
# plt.figure(figsize=(10,6))
# plt.subplot(2,1,1)
# plt.plot(merged_df['Date'], merged_df['Hawkish_Score'], label='Hawkish Score', color='r')
# plt.title('Hawkish Score over Time')
# plt.xlabel('Date')
# plt.ylabel('Hawkish Score')
# plt.legend()

# plt.subplot(2,1,2)
# plt.plot(merged_df['Date'], merged_df['Dovish_Score'], label='Dovish Score', color='b')
# plt.title('Dovish Score over Time')
# plt.xlabel('Date')
# plt.ylabel('Dovish Score')
# plt.legend()

# plt.tight_layout()
# plt.show()
