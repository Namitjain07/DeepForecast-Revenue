"""
Database Manager - MongoDB Connection with Motor (Async)
Handles async database operations for FastAPI
"""

from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import ConnectionFailure
import os
from dotenv import load_dotenv
from loguru import logger
from typing import Optional
from datetime import datetime
from bson import ObjectId

# Load environment variables
load_dotenv()


class DatabaseManager:
    """
    Manages MongoDB connection using Motor async driver
    """
    
    def __init__(self):
        self.client: Optional[AsyncIOMotorClient] = None
        self.db = None
        self.mongodb_uri = os.getenv("MONGODB_URI")
        self.db_name = os.getenv("MONGODB_NAME", "test")  # Default database name
        
    async def connect(self):
        """Connect to MongoDB"""
        try:
            if not self.mongodb_uri:
                raise ValueError("MONGODB_URI not found in environment variables")
            
            self.client = AsyncIOMotorClient(self.mongodb_uri)
            self.db = self.client[self.db_name]
            
            # Test connection
            await self.client.admin.command('ping')
            logger.info(f"Connected to MongoDB database: {self.db_name}")
            
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {str(e)}")
            raise
    
    async def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")
    
    async def health_check(self) -> dict:
        """Check MongoDB connection health"""
        try:
            await self.client.admin.command('ping')
            return {"status": "connected", "database": self.db_name}
        except ConnectionFailure as e:
            logger.error(f"MongoDB health check failed: {str(e)}")
            return {"status": "disconnected", "error": str(e)}
    
    # ============= LastTrain Operations =============
    
    async def create_last_train(self, hotel_id: str, user_id: str) -> dict:
        """
        Create a new LastTrain record
        Returns the created document as a dict
        """
        try:
            last_train_doc = {
                "hotelId": ObjectId(hotel_id),
                "userId": ObjectId(user_id),
                "startDateTime": datetime.utcnow(),
                "endDateTime": datetime.utcnow(),
                "status": "queued",
                "createdAt": datetime.utcnow(),
                "updatedAt": datetime.utcnow()
            }
            
            result = await self.db.lasttrains.insert_one(last_train_doc)
            last_train_doc["_id"] = result.inserted_id
            
            logger.info(f"Created LastTrain record: {result.inserted_id} for hotel: {hotel_id}")
            return self._convert_objectid_to_str(last_train_doc)
            
        except Exception as e:
            logger.error(f"Error creating LastTrain record: {str(e)}")
            raise
    
    async def get_last_train_by_id(self, train_id: str) -> Optional[dict]:
        """Get LastTrain record by ID"""
        try:
            result = await self.db.lasttrains.find_one({"_id": ObjectId(train_id)})
            if result:
                return self._convert_objectid_to_str(result)
            return None
        except Exception as e:
            logger.error(f"Error fetching LastTrain by ID: {str(e)}")
            raise
    
    async def get_last_train_by_hotel(self, hotel_id: str) -> Optional[dict]:
        """Get most recent LastTrain record for a hotel"""
        try:
            result = await self.db.lasttrains.find_one(
                {"hotelId": ObjectId(hotel_id)},
                sort=[("createdAt", -1)]
            )
            if result:
                return self._convert_objectid_to_str(result)
            return None
        except Exception as e:
            logger.error(f"Error fetching LastTrain by hotel: {str(e)}")
            raise
    
    async def update_last_train_status(
        self, 
        train_id: str, 
        status: str, 
        end_datetime: Optional[datetime] = None
    ) -> bool:
        """
        Update LastTrain status and optionally end datetime
        status: 'queued', 'running', 'success', 'failure'
        """
        try:
            update_doc = {
                "status": status,
                "updatedAt": datetime.utcnow()
            }
            
            if end_datetime:
                update_doc["endDateTime"] = end_datetime
            
            result = await self.db.lasttrains.update_one(
                {"_id": ObjectId(train_id)},
                {"$set": update_doc}
            )
            
            logger.info(f"Updated LastTrain {train_id} status to: {status}")
            return result.modified_count > 0
            
        except Exception as e:
            logger.error(f"Error updating LastTrain status: {str(e)}")
            raise
    
    # ============= Record Operations =============
    
    async def get_records_by_hotel(self, hotel_id: str, limit: Optional[int] = None) -> list:
        """
        Fetch all records for a hotel, sorted by date
        Used for training data preparation
        """
        try:
            query = {"hotelId": ObjectId(hotel_id)}
            cursor = self.db.records.find(query).sort("date", 1)
            
            if limit:
                cursor = cursor.limit(limit)
            
            records = []
            async for record in cursor:
                records.append(self._convert_objectid_to_str(record))
            
            logger.info(f"Fetched {len(records)} records for hotel: {hotel_id}")
            return records
            
        except Exception as e:
            logger.error(f"Error fetching records: {str(e)}")
            raise
    
    # ============= Forecast Operations =============
    
    async def save_forecast(
        self, 
        hotel_id: str, 
        model_name: str, 
        predictions: list,
        metadata: dict = None
    ) -> str:
        """
        Save a single model's forecast predictions to database
        
        Args:
            hotel_id: Hotel ID
            model_name: Name of the model (e.g., 'room_revenue', 'arrival_rooms')
            predictions: List of prediction dictionaries with 'ds' and 'value'
            metadata: Additional metadata about the forecast
            
        Returns:
            Inserted document ID as string
        """
        try:
            forecast_doc = {
                "hotelId": ObjectId(hotel_id),
                "modelName": model_name,
                "predictions": predictions,
                "metadata": metadata or {},
                "createdAt": datetime.utcnow(),
                "updatedAt": datetime.utcnow()
            }
            
            result = await self.db.forecasts.insert_one(forecast_doc)
            logger.info(f"Saved forecast for model {model_name}, hotel: {hotel_id}")
            return str(result.inserted_id)
            
        except Exception as e:
            logger.error(f"Error saving forecast: {str(e)}")
            raise
    
    async def save_forecasts(self, hotel_id: str, forecasts: list) -> int:
        """
        Save forecast predictions to database
        Returns count of inserted documents
        """
        try:
            forecast_docs = []
            for forecast in forecasts:
                # Ensure date is a datetime object
                forecast_date = forecast.get("date")
                if not isinstance(forecast_date, datetime):
                    if hasattr(forecast_date, 'to_pydatetime'):
                        forecast_date = forecast_date.to_pydatetime()
                    else:
                        logger.warning(f"Invalid date format: {forecast_date}")
                        continue
                
                forecast_doc = {
                    "hotelId": ObjectId(hotel_id),
                    "date": forecast_date,
                    "revenue": float(forecast.get("revenue", 0)),
                    "roomSold": float(forecast.get("roomSold", 0)),
                    "arrivalRoom": float(forecast.get("arrivalRoom", 0)),
                    "departureRoom": float(forecast.get("departureRoom", 0)),
                    "oooRoom": float(forecast.get("oooRoom", 0))
                }
                forecast_docs.append(forecast_doc)
            
            if forecast_docs:
                result = await self.db.forecasts.insert_many(forecast_docs)
                logger.info(f"Saved {len(result.inserted_ids)} forecasts for hotel: {hotel_id}")
                return len(result.inserted_ids)
            
            return 0
            
        except Exception as e:
            logger.error(f"Error saving forecasts: {str(e)}")
            raise
    
    async def delete_forecasts_by_hotel(self, hotel_id: str) -> int:
        """
        Delete existing forecasts for a hotel before saving new ones
        Returns count of deleted documents
        """
        try:
            query = {"hotelId": ObjectId(hotel_id)}
            result = await self.db.forecasts.delete_many(query)
            logger.info(f"Deleted {result.deleted_count} old forecasts for hotel: {hotel_id}")
            return result.deleted_count
        except Exception as e:
            logger.error(f"Error deleting forecasts: {str(e)}")
            raise
    
    # ============= Utility Methods =============
    
    def _convert_objectid_to_str(self, doc: dict) -> dict:
        """Convert ObjectId fields to strings for JSON serialization"""
        if doc is None:
            return None
        
        converted = {}
        for key, value in doc.items():
            if isinstance(value, ObjectId):
                converted[key] = str(value)
            elif isinstance(value, datetime):
                converted[key] = value.isoformat()
            else:
                converted[key] = value
        
        return converted


# Global instance
db_manager = DatabaseManager()
