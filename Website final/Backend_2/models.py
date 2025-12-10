"""
Pydantic Models
Data validation and serialization models for FastAPI
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from enum import Enum


class TrainStatus(str, Enum):
    """Training job status enum"""
    NONE = "none"
    QUEUED = "queued"
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"


class TrainRequest(BaseModel):
    """Request model for starting training"""
    userId: str = Field(..., description="User ID initiating the training")
    hotelId: str = Field(..., description="Hotel ID for which to train models")
    
    class Config:
        json_schema_extra = {
            "example": {
                "userId": "507f1f77bcf86cd799439011",
                "hotelId": "507f191e810c19729de860ea"
            }
        }


class TrainResponse(BaseModel):
    """Response model for training request"""
    message: str
    job_id: str
    train_id: str
    lastTrain: dict
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "Training job queued successfully",
                "job_id": "550e8400-e29b-41d4-a716-446655440000",
                "train_id": "507f1f77bcf86cd799439011",
                "lastTrain": {
                    "_id": "507f1f77bcf86cd799439011",
                    "hotelId": "507f191e810c19729de860ea",
                    "userId": "507f1f77bcf86cd799439011",
                    "startDateTime": "2024-01-01T12:00:00",
                    "endDateTime": "2024-01-01T12:00:00",
                    "status": "queued"
                }
            }
        }


class JobStatusResponse(BaseModel):
    """Response model for job status query"""
    job_id: str
    train_id: str
    hotel_id: str
    user_id: str
    status: TrainStatus
    queue_position: Optional[int] = None
    error_message: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "550e8400-e29b-41d4-a716-446655440000",
                "train_id": "507f1f77bcf86cd799439011",
                "hotel_id": "507f191e810c19729de860ea",
                "user_id": "507f1f77bcf86cd799439011",
                "status": "running",
                "queue_position": None
            }
        }


class LastTrainResponse(BaseModel):
    """Response model for LastTrain query"""
    message: str
    lastTrain: dict
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "Last train record retrieved successfully",
                "lastTrain": {
                    "_id": "507f1f77bcf86cd799439011",
                    "hotelId": "507f191e810c19729de860ea",
                    "userId": "507f1f77bcf86cd799439011",
                    "startDateTime": "2024-01-01T12:00:00",
                    "endDateTime": "2024-01-01T13:30:00",
                    "status": "success",
                    "createdAt": "2024-01-01T12:00:00",
                    "updatedAt": "2024-01-01T13:30:00"
                }
            }
        }


class ErrorResponse(BaseModel):
    """Standard error response model"""
    message: str
    detail: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "An error occurred",
                "detail": "Detailed error information"
            }
        }


class HealthCheckResponse(BaseModel):
    """Health check response model"""
    status: str
    mongodb: dict
    redis: dict
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "mongodb": {
                    "status": "connected",
                    "database": "test"
                },
                "redis": {
                    "status": "connected",
                    "queue_length": 0
                }
            }
        }


class QueueStatsResponse(BaseModel):
    """Queue statistics response"""
    queue_length: int
    active_jobs: int
    completed_jobs: int
    failed_jobs: int
    
    class Config:
        json_schema_extra = {
            "example": {
                "queue_length": 2,
                "active_jobs": 1,
                "completed_jobs": 15,
                "failed_jobs": 2
            }
        }
