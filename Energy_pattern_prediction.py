import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Create a sample daily energy consumption dataset
np.random.seed(42)  # For reproducible results

# Generate date range for 3 years of daily data
start_date = '2021-01-01'
end_date = '2023-12-31'
date_range = pd.date_range(start=start_date, end=end_date, freq='D')

# Create realistic energy consumption patterns
n_days = len(date_range)

# Base consumption with trend (increasing over time)
base_consumption = 100 + np.linspace(0, 20, n_days)

# Seasonal pattern (higher in summer and winter)
seasonal_pattern = 15 * np.sin(2 * np.pi * np.arange(n_days) / 365.25) + \
                  10 * np.cos(4 * np.pi * np.arange(n_days) / 365.25)

# Weekly pattern (lower on weekends)
weekly_pattern = 5 * np.sin(2 * np.pi * np.arange(n_days) / 7)

# Random noise
noise = np.random.normal(0, 5, n_days)

# Combine all components
energy_consumption = base_consumption + seasonal_pattern + weekly_pattern + noise

# Create DataFrame
energy_data = pd.DataFrame({
    'date': date_range,
    'energy_consumption': energy_consumption
})

# Convert date column to datetime and set as index
energy_data['date'] = pd.to_datetime(energy_data['date'])
energy_data.set_index('date', inplace=True)

# Resample to weekly averages
weekly_data = energy_data.resample('W').mean()
print("Weekly resampled data:")
print(weekly_data.head())

# Resample to monthly averages
monthly_data = energy_data.resample('M').mean()
print("\nMonthly resampled data:")
print(monthly_data.head())

# Create a comparison plot
fig, axes = plt.subplots(3, 1, figsize=(15, 12))

# Daily data
axes[0].plot(energy_data.index, energy_data['energy_consumption'], 
             color='blue', alpha=0.7, linewidth=0.8)
axes[0].set_title('Daily Energy Consumption', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Energy (kWh)')
axes[0].grid(True, alpha=0.3)

# Weekly data
axes[1].plot(weekly_data.index, weekly_data['energy_consumption'], 
             color='green', linewidth=2, marker='o', markersize=3)
axes[1].set_title('Weekly Average Energy Consumption', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Energy (kWh)')
axes[1].grid(True, alpha=0.3)

# Monthly data
axes[2].plot(monthly_data.index, monthly_data['energy_consumption'], 
             color='red', linewidth=3, marker='s', markersize=5)
axes[2].set_title('Monthly Average Energy Consumption', fontsize=14, fontweight='bold')
axes[2].set_ylabel('Energy (kWh)')
axes[2].set_xlabel('Date')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Use additive decomposition (components add up to original series)
decomposition_daily = seasonal_decompose(
    energy_data['energy_consumption'], 
    model='additive', 
    period=365  # Annual seasonality
)

# Apply decomposition to monthly data for clearer seasonal patterns
decomposition_monthly = seasonal_decompose(
    monthly_data['energy_consumption'], 
    model='additive', 
    period=12  # Monthly seasonality (12 months per year)
)

# Extract components from daily decomposition
trend_daily = decomposition_daily.trend
seasonal_daily = decomposition_daily.seasonal
residual_daily = decomposition_daily.resid
observed_daily = decomposition_daily.observed

# Extract components from monthly decomposition
trend_monthly = decomposition_monthly.trend
seasonal_monthly = decomposition_monthly.seasonal
residual_monthly = decomposition_monthly.resid
observed_monthly = decomposition_monthly.observed

# Create comprehensive decomposition plot for daily data
fig, axes = plt.subplots(4, 1, figsize=(16, 12))

# Original time series
axes[0].plot(observed_daily.index, observed_daily.values, color='black', linewidth=1)
axes[0].set_title('Original Daily Energy Consumption Time Series', 
                  fontsize=14, fontweight='bold')
axes[0].set_ylabel('Energy (kWh)')
axes[0].grid(True, alpha=0.3)

# Trend component
axes[1].plot(trend_daily.index, trend_daily.values, color='blue', linewidth=2)
axes[1].set_title('Trend Component (Long-term Pattern)', 
                  fontsize=14, fontweight='bold')
axes[1].set_ylabel('Energy (kWh)')
axes[1].grid(True, alpha=0.3)

# Seasonal component
axes[2].plot(seasonal_daily.index, seasonal_daily.values, color='green', linewidth=1)
axes[2].set_title('Seasonal Component (Yearly Pattern)', 
                  fontsize=14, fontweight='bold')
axes[2].set_ylabel('Energy (kWh)')
axes[2].grid(True, alpha=0.3)

# Residual component
axes[3].plot(residual_daily.index, residual_daily.values, color='red', linewidth=1, alpha=0.7)
axes[3].set_title('Residual Component (Random Variations)', 
                  fontsize=14, fontweight='bold')
axes[3].set_ylabel('Energy (kWh)')
axes[3].set_xlabel('Date')
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Create decomposition plot for monthly data
fig, axes = plt.subplots(4, 1, figsize=(16, 12))

# Original time series
axes[0].plot(observed_monthly.index, observed_monthly.values, 
             color='black', linewidth=2, marker='o', markersize=4)
axes[0].set_title('Original Monthly Energy Consumption Time Series', 
                  fontsize=14, fontweight='bold')
axes[0].set_ylabel('Energy (kWh)')
axes[0].grid(True, alpha=0.3)

# Trend component
axes[1].plot(trend_monthly.index, trend_monthly.values, 
             color='blue', linewidth=3, marker='s', markersize=5)
axes[1].set_title('Trend Component (Long-term Pattern)', 
                  fontsize=14, fontweight='bold')
axes[1].set_ylabel('Energy (kWh)')
axes[1].grid(True, alpha=0.3)

# Seasonal component
axes[2].plot(seasonal_monthly.index, seasonal_monthly.values, 
             color='green', linewidth=2, marker='^', markersize=4)
axes[2].set_title('Seasonal Component (Monthly Pattern)', 
                  fontsize=14, fontweight='bold')
axes[2].set_ylabel('Energy (kWh)')
axes[2].grid(True, alpha=0.3)

# Residual component
axes[3].plot(residual_monthly.index, residual_monthly.values, 
             color='red', linewidth=2, marker='v', markersize=4, alpha=0.7)
axes[3].set_title('Residual Component (Random Variations)', 
                  fontsize=14, fontweight='bold')
axes[3].set_ylabel('Energy (kWh)')
axes[3].set_xlabel('Date')
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Detailed trend analysis
fig, axes = plt.subplots(2, 1, figsize=(15, 10))

# Daily trend
axes[0].plot(trend_daily.index, trend_daily.values, color='blue', linewidth=2)
axes[0].fill_between(trend_daily.index, trend_daily.values, alpha=0.3, color='blue')
axes[0].set_title('Daily Energy Consumption Trend Analysis', 
                  fontsize=14, fontweight='bold')
axes[0].set_ylabel('Energy (kWh)')
axes[0].grid(True, alpha=0.3)

# Add trend statistics
trend_start = trend_daily.dropna().iloc[0]
trend_end = trend_daily.dropna().iloc[-1]
trend_change = trend_end - trend_start
trend_change_percent = (trend_change / trend_start) * 100

axes[0].text(0.02, 0.95, f'Trend Change: {trend_change:.2f} kWh ({trend_change_percent:.1f}%)', 
             transform=axes[0].transAxes, fontsize=12, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

# Monthly trend
axes[1].plot(trend_monthly.index, trend_monthly.values, 
             color='red', linewidth=3, marker='o', markersize=5)
axes[1].fill_between(trend_monthly.index, trend_monthly.values, alpha=0.3, color='red')
axes[1].set_title('Monthly Energy Consumption Trend Analysis', 
                  fontsize=14, fontweight='bold')
axes[1].set_ylabel('Energy (kWh)')
axes[1].set_xlabel('Date')
axes[1].grid(True, alpha=0.3)

# Add monthly trend statistics
monthly_trend_start = trend_monthly.dropna().iloc[0]
monthly_trend_end = trend_monthly.dropna().iloc[-1]
monthly_trend_change = monthly_trend_end - monthly_trend_start
monthly_trend_change_percent = (monthly_trend_change / monthly_trend_start) * 100

axes[1].text(0.02, 0.95, f'Trend Change: {monthly_trend_change:.2f} kWh ({monthly_trend_change_percent:.1f}%)', 
             transform=axes[1].transAxes, fontsize=12, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

plt.tight_layout()
plt.show()

# Seasonal pattern analysis
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Daily seasonal pattern (showing one year)
one_year_seasonal = seasonal_daily['2022-01-01':'2022-12-31']
axes[0, 0].plot(one_year_seasonal.index.dayofyear, one_year_seasonal.values, 
                color='green', linewidth=2)
axes[0, 0].set_title('Daily Seasonal Pattern (One Year)', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Day of Year')
axes[0, 0].set_ylabel('Seasonal Component (kWh)')
axes[0, 0].grid(True, alpha=0.3)

# Monthly seasonal pattern
monthly_seasonal_avg = seasonal_monthly.groupby(seasonal_monthly.index.month).mean()
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
axes[0, 1].bar(months, monthly_seasonal_avg.values, color='orange', alpha=0.7)
axes[0, 1].set_title('Average Monthly Seasonal Pattern', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Seasonal Component (kWh)')
axes[0, 1].tick_params(axis='x', rotation=45)
axes[0, 1].grid(True, alpha=0.3)

# Residual analysis - histogram
axes[1, 0].hist(residual_daily.dropna(), bins=50, color='red', alpha=0.7, edgecolor='black')
axes[1, 0].set_title('Distribution of Daily Residuals', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Residual Value (kWh)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].grid(True, alpha=0.3)

# Residual analysis - time series
axes[1, 1].plot(residual_monthly.index, residual_monthly.values, 
                color='purple', linewidth=1, alpha=0.8)
axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
axes[1, 1].set_title('Monthly Residuals Over Time', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Date')
axes[1, 1].set_ylabel('Residual Value (kWh)')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Create a final summary visualization
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

# Plot all components together for comparison
ax.plot(observed_daily.index, observed_daily.values, 
        label='Original', color='black', linewidth=1, alpha=0.7)
ax.plot(trend_daily.index, trend_daily.values, 
        label='Trend', color='blue', linewidth=2)
ax.plot(observed_daily.index, trend_daily.values + seasonal_daily.values, 
        label='Trend + Seasonal', color='green', linewidth=1.5, alpha=0.8)

ax.set_title('Energy Consumption: Original vs Decomposed Components', 
             fontsize=16, fontweight='bold')
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Energy Consumption (kWh)', fontsize=12)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

'''Troubleshooting Common Issues
Issue 1: Memory or Performance Problems
# If working with very large datasets, consider:
# 1. Sampling the data
sample_data = energy_data.sample(frac=0.1)  # Use 10% of data

# 2. Using different decomposition periods
decomposition_short = seasonal_decompose(
    energy_data['energy_consumption'], 
    model='additive', 
    period=30  # Monthly instead of yearly
)
Issue 2: Missing Values in Decomposition
# Handle missing values before decomposition
energy_data_clean = energy_data.dropna()
# Or interpolate missing values
energy_data_interpolated = energy_data.interpolate(method='linear')
Issue 3: Choosing the Right Decomposition Model
# Compare additive vs multiplicative models
decomp_additive = seasonal_decompose(energy_data['energy_consumption'], model='additive')
decomp_multiplicative = seasonal_decompose(energy_data['energy_consumption'], model='multiplicative')

# Choose based on whether seasonal variations are constant (additive) or proportional (multiplicative)'''