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

    # Display the head and tail of each sheet for initial inspection
    for key in market_data.keys():
        print(key)
        print(market_data[key].head())
        print(market_data[key].tail())
        print("********************")

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

    # Display the merged DataFrame's head, tail, and missing values count
    print(instrument_level.head())
    print(instrument_level.tail())
    print(instrument_level.isna().sum().sum())

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
    print(market_moves.head())

    # Save the final DataFrame to a CSV file
    market_moves.to_csv(processed_mkt_data_path + "/mkt_data_pct_abs_change.csv", index=False)

def perform_market_analysis()-> None:
    load_market_data('data/raw/FOMC_Data_2011_2024.xlsx','data/processed')

if __name__ == "__main__":
    perform_market_analysis()