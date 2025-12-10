"""
API Routes for Model Training
FastAPI endpoints for training job management
"""

from fastapi import APIRouter, HTTPException, status
from loguru import logger

from models import (
    TrainRequest,
    TrainResponse,
    JobStatusResponse,
    LastTrainResponse,
    ErrorResponse,
    QueueStatsResponse
)
from database import db_manager
from redis_queue import redis_queue
import warnings
warnings.filterwarnings("ignore")

router = APIRouter()


@router.post(
    "/train/start",
    response_model=TrainResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Start model training",
    description="Queue a new model training job for a hotel",
    responses={
        201: {"model": TrainResponse},
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def start_training(request: TrainRequest):
    """
    Start a new model training job
    
    - **userId**: User ID initiating the training
    - **hotelId**: Hotel ID for which to train models
    
    Returns training job details and queues the job for processing
    """
    try:
        # Validate inputs
        if not request.userId:
            raise HTTPException(
                status_code=400,
                detail="User ID is required"
            )
        
        if not request.hotelId:
            raise HTTPException(
                status_code=400,
                detail="Hotel ID is required"
            )
        
        logger.info(f"📝 Received training request for hotel: {request.hotelId}")
        
        # Create LastTrain record in MongoDB
        last_train = await db_manager.create_last_train(
            hotel_id=request.hotelId,
            user_id=request.userId
        )
        
        train_id = last_train["_id"]
        
        # Enqueue job in Redis
        job_id = await redis_queue.enqueue_training_job(
            hotel_id=request.hotelId,
            user_id=request.userId,
            train_id=train_id
        )
        
        logger.info(f"✅ Training job queued: {job_id} for hotel: {request.hotelId}")
        
        return TrainResponse(
            message="Training job queued successfully",
            job_id=job_id,
            train_id=train_id,
            lastTrain=last_train
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error starting training: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start training: {str(e)}"
        )


@router.get(
    "/train/status/{job_id}",
    response_model=JobStatusResponse,
    summary="Get job status",
    description="Get the current status of a training job",
    responses={
        200: {"model": JobStatusResponse},
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def get_job_status(job_id: str):
    """
    Get the status of a training job
    
    - **job_id**: Unique identifier for the training job
    
    Returns current status and details of the job
    """
    try:
        logger.info(f"🔍 Fetching status for job: {job_id}")
        
        # Get job status from Redis
        job_status = await redis_queue.get_job_status(job_id)
        
        if not job_status:
            raise HTTPException(
                status_code=404,
                detail=f"Job not found: {job_id}"
            )
        
        return JobStatusResponse(
            job_id=job_status["job_id"],
            train_id=job_status["train_id"],
            hotel_id=job_status["hotel_id"],
            user_id=job_status["user_id"],
            status=job_status["status"],
            error_message=job_status.get("error_message")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error fetching job status: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch job status: {str(e)}"
        )


@router.get(
    "/train/hotel/{hotel_id}",
    response_model=LastTrainResponse,
    summary="Get last training record",
    description="Get the most recent training record for a hotel",
    responses={
        200: {"model": LastTrainResponse},
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def get_last_train_by_hotel(hotel_id: str):
    """
    Get the most recent training record for a hotel
    
    - **hotel_id**: Hotel ID to query
    
    Returns the latest LastTrain record for the specified hotel
    """
    try:
        logger.info(f"🔍 Fetching last train for hotel: {hotel_id}")
        
        # Get last train record from MongoDB
        last_train = await db_manager.get_last_train_by_hotel(hotel_id)
        
        if not last_train:
            raise HTTPException(
                status_code=404,
                detail=f"No training record found for hotel: {hotel_id}"
            )
        
        return LastTrainResponse(
            message="Last train record retrieved successfully",
            lastTrain=last_train
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error fetching last train: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch last train: {str(e)}"
        )


@router.get(
    "/train/queue/stats",
    response_model=QueueStatsResponse,
    summary="Get queue statistics",
    description="Get statistics about the training job queue",
    responses={
        200: {"model": QueueStatsResponse},
        500: {"model": ErrorResponse}
    }
)
async def get_queue_stats():
    """
    Get training queue statistics
    
    Returns current queue length and job counts by status
    """
    try:
        logger.info("📊 Fetching queue statistics")
        
        # Get queue length
        queue_length = await redis_queue.get_queue_length()
        
        # Get all job statuses
        all_jobs = await redis_queue.get_all_job_statuses()
        
        # Count jobs by status
        active_jobs = sum(1 for job in all_jobs if job["status"] == "running")
        completed_jobs = sum(1 for job in all_jobs if job["status"] == "success")
        failed_jobs = sum(1 for job in all_jobs if job["status"] == "failure")
        
        return QueueStatsResponse(
            queue_length=queue_length,
            active_jobs=active_jobs,
            completed_jobs=completed_jobs,
            failed_jobs=failed_jobs
        )
        
    except Exception as e:
        logger.error(f"❌ Error fetching queue stats: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch queue stats: {str(e)}"
        )
