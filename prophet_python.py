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


# ----------- Data Entry Functions -----------

def add_new_data_row(df, date, room_revenue, exogenous_values=None):
    """
    Add a new row of data to the existing dataframe
    
    Parameters:
    -----------
    df : pandas DataFrame
        The existing dataframe with datetime index
    date : str or datetime
        The date for the new entry
    room_revenue : float
        The room revenue value
    exogenous_values : dict, optional
        Dictionary of exogenous variables and their values
        e.g., {'Rooms Sold': 120, 'Occupancy %': 85.5, 'ARR': 150.25, 'Pax': 180}
    
    Returns:
    --------
    pandas DataFrame
        Updated dataframe with new row
    """
    date = pd.to_datetime(date)
    new_row = pd.DataFrame(index=[date])
    new_row['Room Revenue'] = room_revenue
    
    if exogenous_values:
        for col, value in exogenous_values.items():
            new_row[col] = value
    
    updated_df = pd.concat([df, new_row])
    updated_df = updated_df.sort_index()
    return updated_df

def retrain_model_with_new_data(df, exog_cols, model_path=None):
    """
    Retrain the Prophet model with updated data
    
    Parameters:
    -----------
    df : pandas DataFrame
        The dataframe with datetime index
    exog_cols : list
        List of exogenous column names
    model_path : str, optional
        Path to save the retrained model
        
    Returns:
    --------
    tuple
        (Trained Prophet model, prepared prophet data dataframe)
    """
    # Prepare data
    prophet_data = prepare_prophet_data(df, target_col='Room Revenue', exogenous_cols=exog_cols)
    
    # Define model with default good parameters
    # You could optimize again but for quick retraining we'll use decent defaults
    model = Prophet(
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
        holidays_prior_scale=10.0,
        seasonality_mode='multiplicative',
        weekly_seasonality=True,
        yearly_seasonality=True,
        daily_seasonality=False
    )
    
    # Add regressors
    for col in prophet_data.columns:
        if col not in ['ds', 'y']:
            model.add_regressor(col)
    
    # Fit model
    model.fit(prophet_data)
    
    # Save model if path provided
    if model_path:
        save_model(model, model_path)
    
    return model, prophet_data


# ----------- Text File Output Functions -----------

def save_predictions_to_text(predictions_df, filepath):
    """
    Save prediction results to a text file
    
    Parameters:
    -----------
    predictions_df : pandas DataFrame
        DataFrame with prediction results
    filepath : str
        Path to save the text file
    """
    with open(filepath, 'w') as f:
        f.write("Prophet Model Forecast Results\n")
        f.write("==============================\n\n")
        
        # Format the predictions
        for i, row in predictions_df.iterrows():
            date_str = row['ds'].strftime('%Y-%m-%d')
            forecast = row['yhat']
            lower = row['yhat_lower']
            upper = row['yhat_upper']
            
            f.write(f"Date: {date_str}\n")
            f.write(f"Forecast: {forecast:.2f}\n")
            f.write(f"Lower Bound (95%): {lower:.2f}\n")
            f.write(f"Upper Bound (95%): {upper:.2f}\n")
            f.write("------------------------------\n")
    
    print(f"Predictions saved to {filepath}")
    return filepath

def predict_and_save(model, reference_df, dates, output_file):
    """
    Make predictions for specific dates and save to text file
    
    Parameters:
    -----------
    model : Prophet model
        Trained Prophet model
    reference_df : pandas DataFrame
        Reference DataFrame with all required columns
    dates : list or str
        Dates to predict for
    output_file : str
        Path to save the prediction results
        
    Returns:
    --------
    pandas DataFrame
        DataFrame with prediction results
    """
    predictions = predict_for_dates(dates, model=model, reference_df=reference_df)
    save_predictions_to_text(predictions, output_file)
    return predictions


# ----------- Web Interface Helper Functions -----------

def load_data_from_csv(filepath):
    """
    Load data from CSV file for web interface
    
    Parameters:
    -----------
    filepath : str
        Path to CSV file
    
    Returns:
    --------
    pandas DataFrame
        DataFrame with datetime index
    """
    try:
        df = pd.read_csv(filepath, parse_dates=True)
        date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        
        if date_cols:
            df = df.set_index(date_cols[0])
        else:
            # If no obvious date column, assume first column is date
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            
        return df
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

def generate_forecast_periods(model, reference_df, periods=30, output_file=None):
    """
    Generate forecast for specified number of periods and optionally save to file
    
    Parameters:
    -----------
    model : Prophet model
        Trained Prophet model
    reference_df : pandas DataFrame
        Reference DataFrame with all required columns
    periods : int, optional
        Number of periods (days) to forecast
    output_file : str, optional
        Path to save the prediction results
        
    Returns:
    --------
    pandas DataFrame
        DataFrame with prediction results
    """
    future = model.make_future_dataframe(periods=periods)
    future = add_regressors(future, reference_df)
    forecast = model.predict(future)
    
    if output_file:
        save_predictions_to_text(forecast.tail(periods), output_file)
    
    return forecast.tail(periods)

# Example usage for web interface:

# Load existing data
df = load_data_from_csv('hotel_data.csv')

# Add new data row from web form
exog_values = {
    'Rooms Sold': 120,
    'Occupancy %': 85.5,
    'ARR': 150.25,
    'Pax': 180
}
df = add_new_data_row(df, '2023-05-15', 18500.75, exog_values)

# Retrain model with new data
exog_cols = ['Rooms Sold', 'Occupancy %', 'ARR', 'Pax']
model, prophet_data = retrain_model_with_new_data(df, exog_cols, 'updated_model.pkl')

# Generate predictions for specific dates from web form
dates_to_predict = ['2023-06-01', '2023-07-01', '2023-08-01']
predictions = predict_and_save(model, prophet_data, dates_to_predict, 'predictions.txt')

# Generate forecast for next 30 days
forecast_30d = generate_forecast_periods(model, prophet_data, 30, 'forecast_30days.txt')

