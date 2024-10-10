import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load FOMC Scores
fomc_minutes = pd.read_csv(r'data\processed\cosine_sim_H-D-score_meeting_minutes.csv')  # FOMC Meeting Minutes

# Load Market Data
market_data = pd.read_csv(r'data\raw\market_data_1994_2024.csv')  # The 10Y-2Y spread, 1Y Treasury, DXY, Growth-Value Spread

# Convert Date columns to datetime for proper merging
fomc_minutes['Date'] = pd.to_datetime(fomc_minutes['Date'])
market_data['Date'] = pd.to_datetime(market_data['Date']).dt.date  # Remove time part from market data

# Convert FOMC 'Date' to date (remove timestamp if any)
fomc_minutes['Date'] = fomc_minutes['Date'].dt.date

# Merge FOMC scores with market data on Date
merged_df = pd.merge(fomc_minutes, market_data, on='Date', how='left')

# Forward and Backward fill for missing market data
merged_df = merged_df.fillna(method='ffill').fillna(method='bfill')

# Select relevant columns for regression
merged_df = merged_df[['Date', 'Hawkish_Score', 'Dovish_Score', '10Y_Yield', '2Y_Yield', '10Y-2Y_Spread', '1Y_Yield']]

# Define X (independent variables) and Y (dependent variables)
X = merged_df[['10Y-2Y_Spread', '2Y_Yield', '1Y_Yield']]  # Market Variables (excluding DXY and Growth-Value Spread for now)
Y_hawkish = merged_df['Hawkish_Score']  # Dependent Variable: Hawkish Score
Y_dovish = merged_df['Dovish_Score']    # Dependent Variable: Dovish Score

# Standardize the independent variables (X)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Add constant for regression intercept
X_with_const = sm.add_constant(X_scaled)

# Perform regression for Hawkish Score
model_hawkish = sm.OLS(Y_hawkish, X_with_const)
results_hawkish = model_hawkish.fit()
print(results_hawkish.summary())

# Perform regression for Dovish Score
model_dovish = sm.OLS(Y_dovish, X_with_const)
results_dovish = model_dovish.fit()
print(results_dovish.summary())

# Print R-squared scores
print(f"Hawkish R-squared: {results_hawkish.rsquared}")
print(f"Dovish R-squared: {results_dovish.rsquared}")

# Plotting results for visual analysis
plt.figure(figsize=(10,6))
plt.subplot(2,1,1)
plt.plot(merged_df['Date'], merged_df['Hawkish_Score'], label='Hawkish Score', color='r')
plt.title('Hawkish Score over Time')
plt.xlabel('Date')
plt.ylabel('Hawkish Score')
plt.legend()

plt.subplot(2,1,2)
plt.plot(merged_df['Date'], merged_df['Dovish_Score'], label='Dovish Score', color='b')
plt.title('Dovish Score over Time')
plt.xlabel('Date')
plt.ylabel('Dovish Score')
plt.legend()

plt.tight_layout()
plt.show()
