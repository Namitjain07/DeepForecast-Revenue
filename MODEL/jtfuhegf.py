# %%
import matplotlib.pyplot as plt
import pandas as pd

import pandas as pd
import numpy as np
# from prophet import Prophet
# from prophet.diagnostics import cross_validation, performance_metrics
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import holidays
import os
import pickle
import torch
from torch.utils.data import Dataset, DataLoader


# %%
merged_df = pd.read_pickle('df_full.pkl')
merged_df = merged_df.reset_index()


# %%
merged_df[merged_df['Date'] == pd.Timestamp('2023-09-10')]


# %%
print(merged_df.columns.tolist())

# %%
import matplotlib.pyplot as plt
import pandas as pd

# Ensure 'Date' is datetime
merged_df['Date'] = pd.to_datetime(merged_df['Date'])

# Remove outliers using IQR
Q1 = merged_df['Room Revenue'].quantile(0.25)
Q3 = merged_df['Room Revenue'].quantile(0.75)
IQR = Q3 - Q1
filtered_df=merged_df
# Keep only non-outlier data
filtered_df = merged_df[
    (merged_df['Room Revenue'] >= Q1 - 1.5 * IQR) &
    (merged_df['Room Revenue'] <= Q3 + 1.5 * IQR)
]

# Scatter plot with thin dots after outlier removal
# plt.figure(figsize=(10, 6))
# plt.scatter(filtered_df['Date'], filtered_df['Room Revenue'], color='blue', s=10)

# plt.title('Date vs Room Revenue (Outliers Removed)')
# plt.xlabel('Date')
# plt.ylabel('Room Revenue')
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()


# %%
import pandas as pd

# Ensure 'Date' is datetime and sorted
merged_df['Date'] = pd.to_datetime(merged_df['Date'])
merged_df = merged_df.sort_values('Date')

# Create a complete date range
full_range = pd.date_range(merged_df['Date'].min(), merged_df['Date'].max())

# Find missing dates
missing_dates = full_range.difference(merged_df['Date'])

if len(missing_dates) == 0:
    print("✅ No missing dates.")
else:
    print(f"⚠️ Total missing dates: {len(missing_dates)}")
    print("Showing first 20 missing dates (to avoid overload):")
    print(missing_dates)


# %%
merged_df[merged_df['Date'] == pd.Timestamp('2023-09-10')]


# %%
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from neuralprophet import NeuralProphet, set_log_level
from sklearn.metrics import mean_squared_error
import holidays

set_log_level("ERROR")

# =====================================================
# STEP 1: Data Preparation Function (Robust)
# =====================================================
def prepare_prophet_data(df, target_col='Room Revenue', exogenous_cols=None):
    # --- Step 1: Normalize column names ---
    # df.columns = df.columns.str.strip()
    # df.columns = df.columns.str.title()

    # --- Step 2: Handle separate Date and Day columns ---
    # if 'Date' in df.columns and 'Day' in df.columns:
    #     df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    # elif 'date' in df.columns:
    #     df['Date'] = pd.to_datetime(df['date'], errors='coerce')
    # else:
    #     raise KeyError("No column named 'Date' or 'date' found in dataset.")
    
    # df = df.sort_values('Date').drop_duplicates(subset=['Date'])
    df.set_index('Date', inplace=True)

    # --- Step 3: Prophet-format dataset ---
    prophet_df = pd.DataFrame({'ds': df.index, 'y': df[target_col]})
    
    # --- Step 4: Add exogenous regressors ---
    # if exogenous_cols:
    #     for col in exogenous_cols:
    #         col_clean = col.strip()
    #         if col_clean in df.columns:
    # prophet_df['Arrival Rooms'] = df['Arrival Rooms']
    
    # --- Step 5: Add time-based features ---
    # prophet_df['is_weekend'] = prophet_df['ds'].dt.dayofweek.isin([5, 6]).astype(int)
    # prophet_df['month'] = prophet_df['ds'].dt.month
    # prophet_df['quarter'] = prophet_df['ds'].dt.quarter

    # --- Step 6: Add Indian holidays ---
    # in_holidays = holidays.IN()
    # prophet_df['is_holiday'] = prophet_df['ds'].dt.date.apply(lambda x: int(x in in_holidays))
    
    # --- Step 7: Rolling stats on target ---
    # for window in [7, 14]:
    #     prophet_df[f'rolling_mean_{window}d'] = df[target_col].rolling(window).mean()
    #     prophet_df[f'rolling_std_{window}d'] = df[target_col].rolling(window).std()

    # --- Step 8: Fill missing values ---
    
    prophet_df.fillna(method='bfill', inplace=True)
    prophet_df.fillna(method='ffill', inplace=True)

    # --- Step 9: Drop duplicates ---
    prophet_df = prophet_df.drop_duplicates(subset=['ds'])

    return prophet_df


# =====================================================
# STEP 2: Define exogenous columns
# =====================================================
exog_cols = [
    'Arrival Rooms', 'Compliment Rooms', 'House Use',
    'Individual Confirm', 'Arr', 'Departure Rooms'
]

# =====================================================
# STEP 3: Prepare your data
# =====================================================
prophet_df = prepare_prophet_data(merged_df, target_col='Room Revenue', exogenous_cols=exog_cols)

# Add all derived columns to model as regressors
derived_features = [
    'is_weekend', 'month', 'quarter', 'is_holiday',
    'rolling_mean_7d', 'rolling_std_7d',
    'rolling_mean_14d', 'rolling_std_14d'
]

# =====================================================
# STEP 4: Chronological Split
# =====================================================
train_df = prophet_df[prophet_df['ds'] < '2024-04-01']
test_df  = prophet_df[prophet_df['ds'] >= '2024-04-01']


print("=== Train Data Sample ===")
print(train_df.head())
print(train_df.columns)

# =====================================================
# STEP 5: Define NeuralProphet Model
# =====================================================
m = NeuralProphet(
    n_changepoints=10,
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=True,
    n_lags=7,
    n_forecasts=1,
    quantiles=[0.025, 0.5, 0.975]
)
# Register all regressors (manual + derived)
# for col in exog_cols + derived_features:
#     if col in prophet_df.columns:
#         m.add_future_regressor(name=col, normalize=True)
# m.add_future_regressor(name='Arrival Rooms', normalize=True)
# =====================================================
# STEP 6: Train
# =====================================================
metrics = m.fit(train_df, freq='D', validation_df=test_df)

# =====================================================
# STEP 7: Forecast
# =====================================================
forecast_train = m.predict(train_df)
forecast_test = m.predict(test_df)

print("=== Forecast Sample ===")
print(forecast_train.head()) 
print(forecast_train.columns)

# =====================================================
# STEP 8: Evaluate (Fixed Alignment)
# =====================================================
# Align forecast and actual values by date
forecast_test_aligned = forecast_test[['ds', 'yhat1']].merge(
    test_df[['ds', 'y']], on='ds', how='inner'
)
forecast_test_aligned.dropna(subset=['y', 'yhat1'], inplace=True)

mse = mean_squared_error(forecast_test_aligned['y'], forecast_test_aligned['yhat1'])
rmse=np.sqrt(mse)
print(f"✅ Test RMSE (aligned): {rmse:,.2f}")

# =====================================================
# STEP 9: Plot Forecast (Aligned)
# =====================================================
plt.figure(figsize=(14,6))

plt.plot(train_df['ds'], train_df['y'], 'k-', label="Train Actuals")
plt.plot(test_df['ds'], test_df['y'], 'g-', label="Test Actuals")

plt.plot(forecast_train['ds'], forecast_train['yhat1'], 'b--', alpha=0.7, label="Train Forecast")
plt.plot(forecast_test_aligned['ds'], forecast_test_aligned['yhat1'], 'b-', label="Test Forecast")

plt.fill_between(
    forecast_test['ds'],
    forecast_test['yhat1 2.5%'],
    forecast_test['yhat1 97.5%'],
    color='blue', alpha=0.2, label="95% CI"
)

plt.axvline(pd.to_datetime("2024-04-01"), color="red", linestyle="--", label="Train/Test Split")

plt.gca().yaxis.set_major_formatter(
    ticker.FuncFormatter(lambda x, p: f"₹{x:,.0f}")
)

plt.title("Room Revenue Forecast (NeuralProphet + Exogenous + Time-based Features)")
plt.xlabel("Date")
plt.ylabel("Room Revenue")
plt.legend()
plt.tight_layout()
plt.show()


# %%
# Check data types
print(prophet_df.dtypes)

# Show a few recent rows
print(prophet_df.tail(10))

# Check basic stats
print(prophet_df['y'].describe())

# Check if there are any NaN or zero values
print(prophet_df['y'].isna().sum(), "NaN values in y")
print((prophet_df['y'] == 0).sum(), "zero values in y")



# %%


# %%
gds=prophet_df[(prophet_df['ds'] < '2023-10-30') & (prophet_df['ds'] >= '2023-07-01')]
gds.to_csv('gds.csv', index=False)


