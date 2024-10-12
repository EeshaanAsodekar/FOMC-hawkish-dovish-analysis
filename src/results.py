import pandas as pd

def load_market_data(raw_mkt_data_file_path:str,processed_mkt_data_path:str)-> pd.DataFrame:
    # Read all sheet names from the Excel file
    excel_sheets = pd.ExcelFile(raw_mkt_data_file_path).sheet_names

    # Load each sheet into a separate DataFrame, using the sheet names as the DataFrame names
    market_data = {sheet_name: pd.read_excel(raw_mkt_data_file_path, sheet_name=sheet_name) for sheet_name in excel_sheets}

    # Visualize the data in each of the sheets of the raw mkt data excel sheet
    for key in market_data.keys():
        print(key)
        print(market_data[key].head())
        print(market_data[key].tail())
        print("********************")


    # We'll create an empty list to store the dataframes for each instrument.
    instruments = []

    # Iterate through the market_data dictionary to select the relevant price columns
    for key, df in market_data.items():
        # Select the appropriate price column: PX_MID for GT10 and GT2, PX_LAST for others
        if key in ["GT10", "GT2"]:
            price_column = 'PX_MID'
        else:
            price_column = 'PX_LAST'
        
        # Create a new dataframe with Date and the corresponding price column renamed to the instrument name
        df_instrument = df[['Date', price_column]].rename(columns={price_column: key})
        
        # Append the dataframe to the list
        instruments.append(df_instrument)

    # Merge all the dataframes on the 'Date' column
    instrument_level = instruments[0]
    for df in instruments[1:]:
        instrument_level = pd.merge(instrument_level, df, on='Date', how='outer')

    # Display the final dataframe
    print(instrument_level.head())
    print(instrument_level.tail())
    print(instrument_level.isna().sum().sum())

    # 1. Sort the DataFrame by Date in ascending order (from 2012 to 2024)
    instrument_level = instrument_level.sort_values(by='Date')

    # 2. Forward fill NaN values to handle missing data
    instrument_level = instrument_level.fillna(method='ffill')

    # 3. Create dictionaries for the new columns for percentage change and absolute change
    pct_change_columns = {col: f'{col}_pct_change' for col in instrument_level.columns if col != 'Date'}
    abs_change_columns = {col: f'{col}_abs_change' for col in instrument_level.columns if col != 'Date'}

    # Make a copy of the instrument_level dataframe
    market_moves = instrument_level.copy()

    # Calculate the percentage change and absolute change for each column (except 'Date') and add to a new DataFrame
    for col in instrument_level.columns:
        if col != 'Date':
            market_moves[pct_change_columns[col]] = instrument_level[col].pct_change()  # Percentage change
            market_moves[abs_change_columns[col]] = instrument_level[col].diff()  # Absolute change

    # Select only the relevant columns (Date, percentage change, and absolute change columns)
    market_moves = market_moves[['Date'] + list(pct_change_columns.values()) + list(abs_change_columns.values())]

    # Drop rows where any change column has NaN (especially the first row)
    market_moves = market_moves.dropna()

    # Print the new DataFrame (for verification)
    print(market_moves.head())

    # Optionally, save the DataFrame to a CSV file
    market_moves.to_csv(processed_mkt_data_path+"/mkt_data_pct_abs_change.csv", index=False)

def perform_market_analysis()-> None:
    load_market_data('data/raw/FOMC_Data_2011_2024.xlsx','data/processed')

if __name__ == "__main__":
    perform_market_analysis()