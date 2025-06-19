# DeepForecast-Revenue

This notebook (model_train_classification_4_files.ipynb) demonstrates a comprehensive time series forecasting and analysis pipeline for hotel room revenue. It covers classical ARIMA/SARIMAX models, automated parameter selection, exogenous variable extensions, machine learning regressors, neural networks (LSTM/GRU with attention), ensemble approaches, and advanced decomposition methods.

---

## Table of Contents

1. Prerequisites  
2. Data Inputs  
3. Environment Setup  
4. Notebook Workflow  
   4.1. Setup & Data Loading  
   4.2. ARIMA(1,1,1)  
   4.3. 30 Day ARIMA Forecast  
   4.4. SARIMA with Weekly Seasonality  
   4.5. Auto-ARIMA  
   4.6. Extended SARIMAX with Exogenous Variables  
   4.7. SARIMA Grid Search  
   4.8. Time Series Cross Validation  
   4.9. Ridge Regression Forecast  
   4.10. Prophet Model  
   4.11. XGBoost Regressor  
   4.12. LSTM Neural Network  
   4.13. Ensemble Methods  
   4.14. TBATS & Decomposition  
   4.15. VAR Modeling  
   4.16. Attention Enhanced RNNs (BiLSTM/GRU)  
   4.17. Custom Attention Layer  
5. Saving Outputs  
6. References & Links  

---

## Prerequisites

- Python 3.8
- Libraries:
  - numpy, pandas, matplotlib, seaborn  
  - statsmodels, pmdarima  
  - scikit learn, xgboost, prophet  
  - tensorflow, pytorch forecasting (for TFT), pytorch_lightning  
  - tbats, holidays, joblib  

Install via:

pip install numpy pandas matplotlib seaborn statsmodels pmdarima scikit-learn xgboost prophet tensorflow pytorch-lightning pytorch-forecasting tbats holidays joblib

---

## Data Inputs

- df_4_files_combined_no_outliers_for_AR.pkl  
- df_4_files_combined_no_outliers.pkl  

These pickled DataFrames contain daily room revenue and exogenous features. Ensure they are in the notebook’s working directory.

---

## Environment Setup

1. Clone or open the workspace in VS Code.  
2. Activate your conda/virtual env:  
   ```bash
   conda activate btp308   # or your env name
   ```  
3. Launch Jupyter via VS Code’s notebook support or:  
   ```bash
   jupyter notebook model_train_classification_4_files.ipynb
   ```

---

## Notebook Workflow

### 1. Setup & Data Loading
- Import `pandas`, `statsmodels.api`, `numpy`, etc.
- Load and index data from df_4_files_combined_no_outliers_for_AR.pkl.
- Resample to daily frequency.

### 2. ARIMA(1,1,1)
- Fit `ARIMA(..., order=(1,1,1))` on room revenue.  
- View `model_fit.summary()`.

### 3. 30 Day ARIMA Forecast
- `model_fit.get_forecast(steps=30)`  
- Plot historical vs. 30 day forecast with 95% CI.  
- Outputs:  
  - arima_forecasts.png  
  - acf_residuals_ARIMA.png

### 4. SARIMA with Weekly Seasonality
- Fit `SARIMAX(..., seasonal_order=(1,0,1,7))`.  
- Print summary and forecast 30 days.

### 5. Auto ARIMA
- Use `pmdarima.auto_arima(..., seasonal=True, m=7)` to select best (p,d,q)(P,D,Q,7).  
- Plot 30 day forecast.

### 6. Extended SARIMAX with Exogenous Variables
- Load full history from df_4_files_combined_no_outliers.pkl.  
- Define `input_row` for a known future date (`2024-04-01`).  
- Reindex to include that date, ffill/interpolate features.  
- Fit SARIMAX with `exog=...` and forecast next 7, 30, 90, 365 days.

### 7. SARIMA Grid Search
- Function `grid_search_sarima` iterates over `pdq` & seasonal parameters to minimize AIC.

### 8. Time Series Cross Validation
- `time_series_cv(df, exog_columns, order, seasonal_order)` assesses MAE/RMSE/MAPE over rolling test windows.

### 9. Ridge Regression Forecast
- Train `sklearn.linear_model.Ridge` on exogenous features up to known date.  
- Forecast 7/30/90/365 days with static or drifting exogenous patterns.

### 10. Prophet Model
- Format data (`ds`,`y`), add regressors, fit `Prophet()`.  
- Forecast 7 days, plot components.  
- Outputs:  
  - forecast_prophet.png  
  - forecast_prophet_components.png

### 11. XGBoost Regressor
- Create time features (day/week/year, gaps).  
- Train `xgb.XGBRegressor` and quantile models for confidence intervals.  
- Plot 30 day forecast (forecast_xgboost.png).

### 12. LSTM Neural Network
- Scale revenue, build sequences, train a 2 layer LSTM via Keras.  
- Forecast 30 days ahead.  
- Plot: forecast_lstm.png

### 13. Ensemble Methods
- Combine SARIMAX, Ridge, RandomForest in a simple/weighted average.  
- Empirical CI from model error distribution.  
- Plot ensemble: ensemble_forecast.png

### 14. TBATS & Decomposition
- Fit `TBATS(seasonal_periods=[7,30.5])`, forecast 30 days.  
- Decompose weekly/monthly using `sm.tsa.seasonal_decompose`.  
- Outputs:  
  - fitted_tbats.png  
  - decomposition_weekly.png  
  - decomposition_monthly.png

### 15. VAR Modeling
- Build `VAR` on differenced multivariate series (`Room Revenue`, `Rooms Sold`, `ARR`, `Pax`).  
- 1 day forecast and impulse response plots.  
- Outputs:  
  - irf_rooms_sold.png  
  - irf_arr.png

### 16. Attention Enhanced RNNs
- **BiLSTM with Attention**  
- **GRU with Attention**  
- Compare performance, plot losses and 30 day forecasts:
  - bilstm_attention_loss.png  
  - bilstm_attention_forecast.png  
  - gru_attention_loss.png  
  - gru_attention_forecast.png

### 17. Custom Attention Layer
- `AttentionWithContext` custom layer for weight extraction & visualization.  
- Function `visualize_attention()` produces attention_weights.png.

---

## Saving Outputs

All plots are saved to PNG files in the working directory with descriptive names (e.g., 30_day_room_revenue_forecast_with_ci.png, forecast_ridge_1_month.png, etc.). Models and pickles are stored via `joblib.dump()` where applicable (e.g., best_arima_model.pkl).

---
