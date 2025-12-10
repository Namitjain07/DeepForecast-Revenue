"""
Background Worker for Model Training
Processes jobs from Redis queue asynchronously
"""

import asyncio
import sys
import os
from datetime import datetime
from loguru import logger
from dotenv import load_dotenv

from database import db_manager
from redis_queue import redis_queue
from predict import run_all_models

import warnings
warnings.filterwarnings("ignore")

load_dotenv()

# Configure logger
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | {message}",
    level=os.getenv("LOG_LEVEL", "INFO")
)


async def process_training_job(job_data: dict):
    """
    Process a single training job
    
    Args:
        job_data: Dictionary containing job_id, train_id, hotel_id, user_id
    """
    job_id = job_data["job_id"]
    train_id = job_data["train_id"]
    hotel_id = job_data["hotel_id"]
    user_id = job_data["user_id"]
    
    logger.info(f"🎯 Processing training job: {job_id} for hotel: {hotel_id}")
    
    try:
        # Update job status to running
        await redis_queue.set_job_status(job_id, "running", job_data)
        
        # Update LastTrain status to running
        await db_manager.update_last_train_status(train_id, "running")
        
        logger.info(f"🚀 Starting model training for hotel: {hotel_id}")
        
        # Run models in separate thread pool to avoid blocking event loop
        # Since run_all_models is synchronous and CPU-intensive
        loop = asyncio.get_event_loop()
        forecasts = await loop.run_in_executor(
            None,  # Use default executor
            run_all_models,
            hotel_id
        )
        
        logger.info(f"✅ Model training completed for hotel: {hotel_id}")
        
        # Save forecasts to MongoDB
        logger.info(f"💾 Saving forecasts to database for hotel: {hotel_id}")
        
        print("\n\n\n")
        print("====================forecasts======================")
        print(forecasts.keys())
        print(forecasts['room_revenue'])
        
        try:
            # Combine all forecasts by date
            combined_forecasts = {}
            
            # Map model names to schema field names
            model_to_field = {
                'room_revenue': 'revenue',
                'roomsSold': 'roomSold',
                'arrival_rooms': 'arrivalRoom',
                'departure_rooms': 'departureRoom',
                'oooRooms': 'oooRoom'
            }
            
            # Process each model's predictions
            for model_name, forecast_data in forecasts.items():
                if forecast_data and forecast_data.get('predictions'):
                    field_name = model_to_field.get(model_name)
                    if not field_name:
                        logger.warning(f"Unknown model name: {model_name}, skipping")
                        continue
                    
                    predictions = forecast_data['predictions']
                    logger.info(f"Processing {len(predictions)} predictions for {model_name}")
                    
                    for pred in predictions:
                        # Convert Timestamp to datetime string
                        date_key = pred['ds'].strftime('%Y-%m-%d') if hasattr(pred['ds'], 'strftime') else str(pred['ds']).split()[0]
                        
                        if date_key not in combined_forecasts:
                            combined_forecasts[date_key] = {
                                'date': pred['ds'].to_pydatetime() if hasattr(pred['ds'], 'to_pydatetime') else pred['ds'],
                                'revenue': 0,
                                'roomSold': 0,
                                'arrivalRoom': 0,
                                'departureRoom': 0,
                                'oooRoom': 0
                            }
                        
                        # Add the value for this model
                        combined_forecasts[date_key][field_name] = float(pred['value'])
            
            # Convert to list and save
            forecast_list = list(combined_forecasts.values())
            
            if forecast_list:
                logger.info(f"Saving {len(forecast_list)} combined forecast records")
                
                # Delete old forecasts for this hotel first
                deleted_count = await db_manager.delete_forecasts_by_hotel(hotel_id)
                logger.info(f"Deleted {deleted_count} old forecast records")
                
                # Save new forecasts
                saved_count = await db_manager.save_forecasts(hotel_id, forecast_list)
                logger.info(f"✅ Saved {saved_count} forecast records successfully")
            else:
                logger.warning("No forecasts to save")
            
            logger.info(f"✅ All forecasts saved successfully for hotel: {hotel_id}")
        except Exception as save_error:
            logger.error(f"⚠️ Error saving forecasts: {str(save_error)}")
            logger.exception(save_error)
            # Don't fail the job if saving fails, but log it
        
        logger.info(f"📊 Generated forecasts for {len(forecasts)} models")
        
        # Update job status to success
        end_datetime = datetime.utcnow()
        await redis_queue.set_job_status(
            job_id, 
            "success", 
            {**job_data, "completed_at": end_datetime.isoformat()}
        )
        
        # Update LastTrain status to success
        await db_manager.update_last_train_status(train_id, "success", end_datetime)
        
        logger.info(f"🎉 Training job completed successfully: {job_id}")
        
    except Exception as e:
        logger.error(f"❌ Training job failed: {job_id} - {str(e)}")
        logger.exception(e)
        
        # Update job status to failure
        end_datetime = datetime.utcnow()
        await redis_queue.set_job_status(
            job_id,
            "failure",
            {
                **job_data,
                "error_message": str(e),
                "completed_at": end_datetime.isoformat()
            }
        )
        
        # Update LastTrain status to failure
        await db_manager.update_last_train_status(train_id, "failure", end_datetime)


async def worker_loop():
    """
    Main worker loop
    Continuously processes jobs from the Redis queue
    """
    logger.info("👷 Worker started, waiting for jobs...")
    
    while True:
        try:
            # Block and wait for a job (timeout after 5 seconds to allow graceful shutdown)
            job_data = await redis_queue.dequeue_training_job(timeout=5)
            
            if job_data:
                logger.info(f"📥 Received job: {job_data['job_id']}")
                await process_training_job(job_data)
            else:
                # Timeout reached, no job available
                # This allows the loop to check for shutdown signals
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("⚠️ Worker interrupted by user")
            break
        except Exception as e:
            logger.error(f"❌ Worker error: {str(e)}")
            logger.exception(e)
            # Wait a bit before retrying to avoid rapid error loops
            await asyncio.sleep(5)


async def main():
    """Main entry point for the worker"""
    try:
        # Connect to MongoDB
        await db_manager.connect()
        logger.info("✅ MongoDB connection established")
        
        # Connect to Redis
        await redis_queue.connect()
        logger.info("✅ Redis connection established")
        
        # Start worker loop
        await worker_loop()
        
    except KeyboardInterrupt:
        logger.info("🛑 Worker shutting down...")
    except Exception as e:
        logger.error(f"❌ Worker startup failed: {str(e)}")
        logger.exception(e)
        sys.exit(1)
    finally:
        # Cleanup
        await redis_queue.close()
        await db_manager.close()
        logger.info("👋 Worker stopped")


if __name__ == "__main__":
    logger.info("🚀 Starting Background Worker for Model Training")
    asyncio.run(main())
