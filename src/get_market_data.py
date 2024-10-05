import yfinance as yf
import pandas as pd

def download_market_data(start_date, end_date):
    # Download 10Y and 2Y Treasury bond yields (Calculate Spread)
    print("Downloading 10Y and 2Y Treasury data...")
    treasury_10y = yf.Ticker("^TNX")  # 10Y yield
    treasury_2y = yf.Ticker("^IRX")   # 2Y yield
    treasury_10y_hist = treasury_10y.history(start=start_date, end=end_date)
    treasury_2y_hist = treasury_2y.history(start=start_date, end=end_date)

    treasury_10y_hist = treasury_10y_hist[['Close']].rename(columns={'Close': '10Y_Yield'})
    treasury_2y_hist = treasury_2y_hist[['Close']].rename(columns={'Close': '2Y_Yield'})

    # Align the dataframes and calculate the 10Y-2Y spread
    treasury_spread = pd.merge(treasury_10y_hist, treasury_2y_hist, left_index=True, right_index=True)
    treasury_spread['10Y-2Y_Spread'] = treasury_spread['10Y_Yield'] - treasury_spread['2Y_Yield']

    # Download 1Y Treasury bond yield
    print("Downloading 1Y Treasury data...")
    treasury_1y = yf.Ticker("^FVX")  # 1Y Treasury yield (use an appropriate ticker if needed)
    treasury_1y_hist = treasury_1y.history(start=start_date, end=end_date)[['Close']]
    treasury_1y_hist = treasury_1y_hist.rename(columns={'Close': '1Y_Yield'})

    # Download DXY (Dollar Index)
    print("Downloading DXY (Dollar Index) data...")
    dxy = yf.Ticker("DX-Y.NYB")  # DXY Index
    dxy_hist = dxy.history(start=start_date, end=end_date)[['Close']]
    dxy_hist = dxy_hist.rename(columns={'Close': 'DXY_Index'})

    # Download Growth and Value ETFs (Proxies: IWF for Growth, IWD for Value)
    print("Downloading Growth and Value ETF data...")
    growth_etf = yf.Ticker("IWF")  # iShares Russell 1000 Growth ETF
    value_etf = yf.Ticker("IWD")   # iShares Russell 1000 Value ETF
    growth_hist = growth_etf.history(start=start_date, end=end_date)[['Close']].rename(columns={'Close': 'Growth_ETF'})
    value_hist = value_etf.history(start=start_date, end=end_date)[['Close']].rename(columns={'Close': 'Value_ETF'})

    # Calculate Growth-Value Spread
    growth_value_spread = pd.merge(growth_hist, value_hist, left_index=True, right_index=True)
    growth_value_spread['Growth-Value_Spread'] = growth_value_spread['Growth_ETF'] - growth_value_spread['Value_ETF']

    # Merge all data into a single DataFrame based on Date
    print("Merging all data...")
    merged_data = treasury_spread.merge(treasury_1y_hist, left_index=True, right_index=True, how='left')
    merged_data = merged_data.merge(dxy_hist, left_index=True, right_index=True, how='left')
    merged_data = merged_data.merge(growth_value_spread[['Growth-Value_Spread']], left_index=True, right_index=True, how='left')

    return merged_data

if __name__ == "__main__":
    # Assuming the data ranges from 1994 to 2024 (adjust based on your data)
    start_date = "1994-01-01"
    end_date = "2024-12-31"
    
    # Download and save the data to CSV
    market_data = download_market_data(start_date, end_date)
    market_data.to_csv("data/raw/market_data_1994_2024.csv", index=True)
    print("Market data downloaded and saved to market_data_1994_2024.csv")
