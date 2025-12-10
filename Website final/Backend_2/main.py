"""
FastAPI Model Controller - Main Application
Handles async training requests with Redis queue for DeepForecast-Revenue
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv
from loguru import logger
import sys

from database import db_manager
from redis_queue import redis_queue
from routes import router

# Load environment variables
load_dotenv()

# Configure logger
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | {message}",
    level=os.getenv("LOG_LEVEL", "INFO")
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle manager for FastAPI application
    Handles startup and shutdown events
    """
    # Startup
    logger.info("🚀 Starting FastAPI Model Controller...")
    
    # Connect to MongoDB
    await db_manager.connect()
    logger.info("✅ MongoDB connection established")
    
    # Connect to Redis
    await redis_queue.connect()
    logger.info("✅ Redis connection established")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down FastAPI Model Controller...")
    
    # Close Redis connection
    await redis_queue.close()
    logger.info("✅ Redis connection closed")
    
    # Close MongoDB connection
    await db_manager.close()
    logger.info("✅ MongoDB connection closed")


# Initialize FastAPI app
app = FastAPI(
    title="DeepForecast Model Controller",
    description="Async model training API with Redis queue for hotel revenue forecasting",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(router, prefix="/api/v1")


@app.get("/")
async def root():
    """Root endpoint - health check"""
    return {
        "service": "DeepForecast Model Controller",
        "status": "running",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint
    Verifies MongoDB and Redis connections
    """
    try:
        # Check MongoDB
        mongo_status = await db_manager.health_check()
        
        # Check Redis
        redis_status = await redis_queue.health_check()
        
        return {
            "status": "healthy",
            "mongodb": mongo_status,
            "redis": redis_status
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))
    reload = os.getenv("API_RELOAD", "False").lower() == "true"
    
    logger.info(f"🌐 Starting server on {host}:{port}")
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload
    )
