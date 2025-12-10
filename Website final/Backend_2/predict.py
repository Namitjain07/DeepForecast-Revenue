# predict.py

import sys
import json
import pandas as pd
import numpy as np
import holidays
from neuralprophet import NeuralProphet, set_log_level
from datetime import datetime, timedelta
from loguru import logger
from pymongo import MongoClient
from bson import ObjectId
import os
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")

load_dotenv()

set_log_level("ERROR")

# Configure logger
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | {message}",
    level="INFO"
)


# ----------- MongoDB Data Loading -----------

def load_data_from_mongodb(hotel_id: str) -> pd.DataFrame:
    """
    Load hotel records from MongoDB and convert to dataframe
    
    Args:
        hotel_id: Hotel ID to fetch records for
        
    Returns:
        DataFrame with date as index and relevant columns
    """
    try:
        mongodb_uri = os.getenv("MONGODB_URI")
        db_name = os.getenv("MONGODB_NAME", "test")
        
        if not mongodb_uri:
            raise ValueError("MONGODB_URI not found in environment variables")
        
        logger.info(f"Connecting to MongoDB for hotel: {hotel_id}")
        
        # Connect to MongoDB
        client = MongoClient(mongodb_uri)
        db = client[db_name]
        
        # Fetch records for the hotel
        records = list(db.records.find({"hotelId": ObjectId(hotel_id)}).sort("date", 1))
        
        if not records:
            raise ValueError(f"No records found for hotel: {hotel_id}")
        
        logger.info(f"Fetched {len(records)} records from MongoDB")
        
        # Convert to DataFrame
        df = pd.DataFrame(records)
        
        # Rename columns to match expected format
        column_mapping = {
            'date': 'date',
            'roomRevenue': 'Room Revenue',
            'arrivalRooms': 'Arrival Rooms',
            'departureRooms': 'Departure Rooms',
            'complimentRooms': 'Compliment Rooms',
            'houseUse': 'House Use',
            'roomsSold': 'Rooms Sold',
            'occupancyPercentage': 'Occupancy %',
            'averageRoomRate': 'ARR',
            'oooRooms': 'OOO Rooms',
            'pax': 'PAX',
            'totalRoomInventory': 'Total Room Inventory'
        }
        
        # Rename columns that exist
        df.rename(columns=column_mapping, inplace=True)
        
        # Set date as index
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)
        
        print(df.columns)
        print(df.head())
        
        # Select only the columns we need for training
        required_columns = ['Room Revenue', 'Arrival Rooms', 'Departure Rooms', 'OOO Rooms', 'Rooms Sold']
        existing_columns = [col for col in required_columns if col in df.columns]
        
        if not existing_columns:
            raise ValueError("No required columns found in the data")
        
        df = df[existing_columns]
        
        print('\n\n\n')
        # print(df.columns())
        print(df.head())
        
        logger.info(f"Data loaded: {len(df)} records, date range: {df.index.min()} to {df.index.max()}")
        logger.info(f"Available columns: {df.columns.tolist()}")
        
        # Close MongoDB connection
        client.close()
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading data from MongoDB: {str(e)}")
        raise


# ----------- Data Preparation -----------
def prepare_df_for_neural_prophet(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    df = df.copy()

    # Ensure index is datetime
    df.index = pd.to_datetime(df.index)

    # Remove time portion (normalize to YYYY-MM-DD)
    df.index = df.index.normalize()   # same as .dt.floor("D")

    # Create NeuralProphet-friendly dataframe
    np_df = pd.DataFrame({
        "ds": df.index,
        "y": df[target_col].values
    })
    
    return np_df

def train_neural_prophet_for_target(df: pd.DataFrame, target_col: str, forecast_days: int = 365) -> pd.DataFrame:
    
    # Prepare dataframe
    np_df = prepare_df_for_neural_prophet(df, target_col)
    
    print("\n\n\n")
    print("====================np_df======================")
    print(np_df.head())
    print(np_df.columns)

    # your neural prophet training code…
    model = NeuralProphet(
        n_changepoints=10,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        n_forecasts=1,
        quantiles=[0.025, 0.5, 0.975]
    )
    model.fit(np_df, freq="D")

    future = model.make_future_dataframe(np_df, periods=forecast_days)
    forecast = model.predict(future)
    
    print("\n\n\n")
    print("====================forecast======================")
    print(forecast.head())
    print(forecast.columns)

    return forecast



def run_all_models(hotel_id: str):
    """
    Run ALL 5 DL models and return a dict of forecasts.
    If anything fails, let the exception propagate (main.py will handle it).
    
    Args:
        hotel_id: Hotel ID to fetch data from MongoDB
    
    Returns:
        Dictionary with forecast data for all 5 models
    """
    logger.info(f"🔄 Starting model training for hotel_id={hotel_id}")
    
    # Load data from MongoDB instead of pickle file
    df = load_data_from_mongodb(hotel_id)
    print("\n\n\n")
    print("====================df after load======================")
    print(df.columns)
    print(df.head())
    
    
    logger.info(f"📊 Loaded data: {len(df)} records, date range: {df.index.min()} to {df.index.max()}")

    # Define your 5 targets
    # Adjust target_col names to match columns in df
    models_config = [
        {"name": "room_revenue",        "target_col": "Room Revenue"},
        {"name": "arrival_rooms",       "target_col": "Arrival Rooms"},
        {"name": "departure_rooms",     "target_col": "Departure Rooms"},
        {"name": "oooRooms",    "target_col": "OOO Rooms"},
        {"name": "roomsSold",     "target_col": "Rooms Sold"},
    ]

    results = {}
    failed_models = []
    
    # Check if we have enough data overall
    if len(df) < 14:
        raise ValueError(
            f"Insufficient historical data: {len(df)} records found. "
            f"Need at least 14 days of data for training. Please add more records to this hotel."
        )

    for idx, cfg in enumerate(models_config, 1):
        logger.info(f"🤖 Training model {idx}/5: {cfg['name']} ({cfg['target_col']})")
        
        try:
            # Check if column exists in dataframe
            if cfg["target_col"] not in df.columns:
                logger.warning(f"⚠️ Column {cfg['target_col']} not found in data, skipping...")
                failed_models.append(cfg['name'])
                continue
            
            # Check if column has valid data (not all NaN or zeros)
            if df[cfg["target_col"]].isna().all() or (df[cfg["target_col"]] == 0).all():
                logger.warning(f"⚠️ Column {cfg['target_col']} has no valid data, skipping...")
                failed_models.append(cfg['name'])
                continue
            print("\n\n\n")
            print("====================df======================")
            print(df.columns)
            print(df.head())
                
            forecast_df = train_neural_prophet_for_target(df, cfg["target_col"], forecast_days=365)
            
            
            # Convert to JSON/Mongo friendly format
            # Filter to only future predictions (after last historical date)
            future_forecast = forecast_df[forecast_df['ds'] > df.index.max()].copy()
            
            results[cfg["name"]] = {
                "predictions": future_forecast[['ds', 'yhat1']].rename(columns={'yhat1': 'value'}).to_dict(orient="records"),
                "last_train_date": df.index.max().isoformat(),
                "forecast_start": future_forecast['ds'].min().isoformat() if len(future_forecast) > 0 else None,
                "forecast_end": future_forecast['ds'].max().isoformat() if len(future_forecast) > 0 else None,
                "num_predictions": len(future_forecast)
            }
            
            logger.info(f"✅ Model {idx}/5 completed: {cfg['name']} - {len(future_forecast)} predictions")
        except Exception as e:
            logger.error(f"❌ Model {idx}/5 failed: {cfg['name']} - {str(e)}")
            failed_models.append(cfg['name'])
            # Continue with other models instead of failing completely
            continue

    if len(results) == 0:
        raise ValueError(
            f"All models failed to train successfully. "
            f"Please check data quality and ensure sufficient historical records."
        )
    
    if failed_models:
        logger.warning(f"⚠️ Some models failed: {', '.join(failed_models)}")
    
    logger.info(f"🎉 Training completed! {len(results)}/{len(models_config)} models trained successfully")
    return results


def main():
    """
    Main entry point when called as a subprocess
    Expects: python predict.py <hotel_id> <output_file>
    """
    if len(sys.argv) < 3:
        logger.error("❌ Usage: python predict.py <hotel_id> <output_file>")
        sys.exit(1)
    
    hotel_id = sys.argv[1]
    output_file = sys.argv[2]
    
    logger.info(f"🚀 Starting prediction for hotel_id={hotel_id}")
    logger.info(f"📁 Output will be saved to: {output_file}")
    
    try:
        # Run all models
        forecasts = run_all_models(hotel_id=hotel_id)
        
        print("\n\n\n")
        print("====================forecasts======================")
        print(forecasts.columns)
        print(forecasts.head())
        
        # Save results to output file
        with open(output_file, 'w') as f:
            json.dump(forecasts, f, indent=2)
        
        logger.info(f"💾 Results saved to {output_file}")
        logger.info("✅ Prediction completed successfully")
        
        # Exit with success code
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"❌ Prediction failed: {str(e)}")
        logger.exception(e)
        
        # Exit with failure code
        sys.exit(1)


# Optional: allow running from CLI for debugging
if __name__ == "__main__":
    main()
