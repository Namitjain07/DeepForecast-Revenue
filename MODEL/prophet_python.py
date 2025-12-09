import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import holidays
import os
import pickle

# ----------- Data Preparation -----------

def prepare_prophet_data(df, target_col='Room Revenue', exogenous_cols=None):
    prophet_df = pd.DataFrame({'ds': df.index, 'y': df[target_col]})
    
    # Add exogenous regressors
    if exogenous_cols:
        for col in exogenous_cols:
            if col in df.columns:
                prophet_df[col] = df[col]
    
    # Day of week (one-hot)
    for i in range(7):
        prophet_df[f'day_{i}'] = (prophet_df['ds'].dt.dayofweek == i).astype(int)
    
    # Weekend flag
    prophet_df['is_weekend'] = prophet_df['ds'].dt.dayofweek.isin([5, 6]).astype(int)
    
    # Month and quarter
    prophet_df['month'] = prophet_df['ds'].dt.month
    prophet_df['quarter'] = prophet_df['ds'].dt.quarter
    
    # US holidays
    us_holidays = holidays.US()
    prophet_df['is_holiday'] = (prophet_df['ds'].dt.date.apply(lambda x: x in us_holidays)).astype(int)
    
    # Rolling stats on target (7 and 14 days)
    if not pd.isna(df[target_col]).all():
        for window in [7, 14]:
            prophet_df[f'rolling_mean_{window}d'] = df[target_col].rolling(window=window).mean().values
            prophet_df[f'rolling_std_{window}d'] = df[target_col].rolling(window=window).std().values
    
    prophet_df.fillna(method='bfill', inplace=True)
    return prophet_df


# ----------- Save/Load Model -----------

def save_model(model, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filepath}")

def load_model(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No model file found at {filepath}")
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded from {filepath}")
    return model


# ----------- Prediction Functions -----------

def predict_for_dates(dates, model, reference_df):
    if isinstance(dates, (str, pd.Timestamp)):
        dates = [dates]
    dates = pd.to_datetime(dates)
    
    future_dates = pd.DataFrame({'ds': dates})
    us_holidays = holidays.IN()
    
    for col in reference_df.columns:
        if col not in ['ds', 'y']:
            if col.startswith('day_'):
                day_num = int(col.split('_')[1])
                future_dates[col] = (future_dates['ds'].dt.dayofweek == day_num).astype(int)
            elif col == 'is_weekend':
                future_dates[col] = future_dates['ds'].dt.dayofweek.isin([5,6]).astype(int)
            elif col == 'month':
                future_dates[col] = future_dates['ds'].dt.month
            elif col == 'quarter':
                future_dates[col] = future_dates['ds'].dt.quarter
            elif col == 'is_holiday':
                future_dates[col] = (future_dates['ds'].dt.date.apply(lambda x: x in us_holidays)).astype(int)
            else:
                future_dates[col] = reference_df[col].iloc[-1]
    
    forecast = model.predict(future_dates)
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]


def predict_with_loaded_model(filepath, dates, reference_df):
    model = load_model(filepath)
    return predict_for_dates(dates, model=model, reference_df=reference_df)


# ----------- Hyperparameter Optimization -----------

def objective(params):
    model = Prophet(
        changepoint_prior_scale=params['changepoint_prior_scale'],
        seasonality_prior_scale=params['seasonality_prior_scale'],
        holidays_prior_scale=params['holidays_prior_scale'],
        seasonality_mode=params['seasonality_mode']
    )
    
    for col in prophet_data.columns:
        if col not in ['ds', 'y']:
            model.add_regressor(col)
    
    model.fit(prophet_data)
    
    df_cv = cross_validation(
        model, 
        initial='366 days',
        period='30 days',
        horizon='30 days',
        parallel="processes"
    )
    
    df_p = performance_metrics(df_cv)
    rmse = df_p['rmse'].mean()
    
    return {'loss': rmse, 'status': STATUS_OK}


# ----------- Helper function to add regressors to future dfs -----------

def add_regressors(future_df, reference_df):
    us_holidays = holidays.US()
    for col in reference_df.columns:
        if col not in ['ds', 'y']:
            if col.startswith('day_'):
                day_num = int(col.split('_')[1])
                future_df[col] = (future_df['ds'].dt.dayofweek == day_num).astype(int)
            elif col == 'is_weekend':
                future_df[col] = future_df['ds'].dt.dayofweek.isin([5, 6]).astype(int)
            elif col == 'month':
                future_df[col] = future_df['ds'].dt.month
            elif col == 'quarter':
                future_df[col] = future_df['ds'].dt.quarter
            elif col == 'is_holiday':
                future_df[col] = (future_df['ds'].dt.date.apply(lambda x: x in us_holidays)).astype(int)
            else:
                future_df[col] = reference_df[col].iloc[-1]
    return future_df


# ----------- Plot function -----------

def plot_forecast(forecast_df, title):
    plt.figure(figsize=(12,6))
    plt.plot(prophet_data['ds'], prophet_data['y'], 'k.', label='Historical')
    plt.plot(forecast_df['ds'], forecast_df['yhat'], 'b-', label='Forecast')
    plt.fill_between(forecast_df['ds'], forecast_df['yhat_lower'], forecast_df['yhat_upper'], color='blue', alpha=0.2, label='95% CI')

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.gcf().autofmt_xdate()

    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Room Revenue')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ----------- Main Execution -----------

# Your dataframe with datetime index and data - replace with your actual df
# Example:
# df_4_files_combined_no_outliers_for_AR = pd.read_csv('your_data.csv', parse_dates=['date_column'], index_col='date_column')

exog_cols = ['Rooms Sold', 'Occupancy %', 'ARR', 'Pax']  # example exogenous columns

prophet_data = prepare_prophet_data(df_4_files_combined_no_outliers_for_AR, target_col='Room Revenue', exogenous_cols=exog_cols)

space = {
    'changepoint_prior_scale': hp.loguniform('changepoint_prior_scale', -3, 0),
    'seasonality_prior_scale': hp.loguniform('seasonality_prior_scale', -2, 1),
    'holidays_prior_scale': hp.loguniform('holidays_prior_scale', -2, 0),
    'seasonality_mode': hp.choice('seasonality_mode', ['additive', 'multiplicative'])
}

trials = Trials()
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=30,
            trials=trials)

best_params = {
    'changepoint_prior_scale': np.exp(best['changepoint_prior_scale']),
    'seasonality_prior_scale': np.exp(best['seasonality_prior_scale']),
    'holidays_prior_scale': np.exp(best['holidays_prior_scale']),
    'seasonality_mode': ['additive', 'multiplicative'][best['seasonality_mode']]
}
print("Best parameters found:", best_params)

final_model = Prophet(
    changepoint_prior_scale=best_params['changepoint_prior_scale'],
    seasonality_prior_scale=best_params['seasonality_prior_scale'],
    holidays_prior_scale=best_params['holidays_prior_scale'],
    seasonality_mode=best_params['seasonality_mode'],
    weekly_seasonality=True,
    yearly_seasonality=True,
    daily_seasonality=False
)

for col in prophet_data.columns:
    if col not in ['ds', 'y']:
        final_model.add_regressor(col)

final_model.fit(prophet_data)

# Save final model
save_model(final_model, 'final_model.pkl')


# ----------- Forecast for next 1 month and 3 months -----------

future_1m = final_model.make_future_dataframe(periods=30)
future_3m = final_model.make_future_dataframe(periods=90)

future_1m = add_regressors(future_1m, prophet_data)
future_3m = add_regressors(future_3m, prophet_data)

forecast_1m = final_model.predict(future_1m)
forecast_3m = final_model.predict(future_3m)

# ----------- Forecast for next 1 year -----------

future_1y = final_model.make_future_dataframe(periods=365)
future_1y = add_regressors(future_1y, prophet_data)
forecast_1y = final_model.predict(future_1y)


# ----------- Predict for specific dates -----------

specific_dates = ['2025-06-20', '2025-07-01', '2025-09-15']
pred_specific = predict_for_dates(specific_dates, model=final_model, reference_df=prophet_data)
print("Predictions for specific dates:")
print(pred_specific)


# ----------- Plot forecasts -----------

plot_forecast(forecast_1m, '1-Month Forecast (Next 30 Days)')
plot_forecast(forecast_3m, '3-Month Forecast (Next 90 Days)')
plot_forecast(forecast_1y, '1-Year Forecast (Next 365 Days)')




specific_dates = ['2025-06-20', '2025-07-01', '2025-09-15', '2026-06-15']
pred_specific = predict_for_dates(specific_dates, model=final_model, reference_df=prophet_data)
print("Predictions for specific dates:")
print(pred_specific)

