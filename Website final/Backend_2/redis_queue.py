"""
Redis Queue Manager
Handles job queue for async model training using Redis
"""

import redis.asyncio as aioredis
from redis.asyncio import Redis
import json
import os
from dotenv import load_dotenv
from loguru import logger
from typing import Optional, Dict, Any
import uuid

# Load environment variables
load_dotenv()


class RedisQueue:
    """
    Manages Redis connection and job queue operations
    """
    
    def __init__(self):
        self.redis: Optional[Redis] = None
        self.redis_url = os.getenv("REDIS_URL")
        self.queue_name = "model_training_queue"
        self.status_prefix = "job_status:"
        
    async def connect(self):
        """Connect to Redis"""
        try:
            if not self.redis_url:
                raise ValueError("REDIS_URL not found in environment variables")
            
            self.redis = await aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            
            # Test connection
            await self.redis.ping()
            logger.info("Connected to Redis successfully")
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {str(e)}")
            raise
    
    async def close(self):
        """Close Redis connection"""
        if self.redis:
            await self.redis.close()
            logger.info("Redis connection closed")
    
    async def health_check(self) -> dict:
        """Check Redis connection health"""
        try:
            pong = await self.redis.ping()
            queue_length = await self.get_queue_length()
            return {
                "status": "connected" if pong else "disconnected",
                "queue_length": queue_length
            }
        except Exception as e:
            logger.error(f"Redis health check failed: {str(e)}")
            return {"status": "disconnected", "error": str(e)}
    
    async def enqueue_training_job(
        self, 
        hotel_id: str, 
        user_id: str, 
        train_id: str
    ) -> str:
        """
        Add a training job to the Redis queue
        Returns the job_id
        """
        try:
            job_id = str(uuid.uuid4())
            
            job_data = {
                "job_id": job_id,
                "train_id": train_id,
                "hotel_id": hotel_id,
                "user_id": user_id,
                "status": "queued"
            }
            
            # Push job to queue (LPUSH adds to left, workers pop from right with BRPOP)
            await self.redis.lpush(self.queue_name, json.dumps(job_data))
            
            # Store job status separately for quick lookup
            await self.set_job_status(job_id, "queued", job_data)
            
            logger.info(f"Enqueued training job: {job_id} for hotel: {hotel_id}")
            return job_id
            
        except Exception as e:
            logger.error(f"Error enqueuing training job: {str(e)}")
            raise
    
    async def dequeue_training_job(self, timeout: int = 0) -> Optional[Dict[str, Any]]:
        """
        Dequeue a training job from Redis (blocking pop)
        Used by workers to fetch jobs
        timeout: 0 = blocking forever, >0 = timeout in seconds
        """
        try:
            # BRPOP: blocking right pop (waits until item available)
            result = await self.redis.brpop(self.queue_name, timeout=timeout)
            
            if result:
                _, job_json = result
                job_data = json.loads(job_json)
                logger.info(f"Dequeued training job: {job_data['job_id']}")
                return job_data
            
            return None
            
        except Exception as e:
            logger.error(f"Error dequeuing training job: {str(e)}")
            raise
    
    async def get_queue_length(self) -> int:
        """Get current queue length"""
        try:
            length = await self.redis.llen(self.queue_name)
            return length
        except Exception as e:
            logger.error(f"Error getting queue length: {str(e)}")
            return 0
    
    async def set_job_status(
        self, 
        job_id: str, 
        status: str, 
        data: Optional[Dict[str, Any]] = None
    ):
        """
        Update job status in Redis
        Stored with TTL (time to live) of 24 hours
        """
        try:
            status_key = f"{self.status_prefix}{job_id}"
            
            status_data = {
                "job_id": job_id,
                "status": status
            }
            
            if data:
                status_data.update(data)
            
            # Store with 24-hour expiration
            await self.redis.setex(
                status_key,
                86400,  # 24 hours in seconds
                json.dumps(status_data)
            )
            
            logger.debug(f"Updated job status: {job_id} -> {status}")
            
        except Exception as e:
            logger.error(f"Error setting job status: {str(e)}")
            raise
    
    async def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get job status from Redis
        Returns None if job not found or expired
        """
        try:
            status_key = f"{self.status_prefix}{job_id}"
            status_json = await self.redis.get(status_key)
            
            if status_json:
                return json.loads(status_json)
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting job status: {str(e)}")
            raise
    
    async def delete_job_status(self, job_id: str):
        """Delete job status from Redis"""
        try:
            status_key = f"{self.status_prefix}{job_id}"
            await self.redis.delete(status_key)
            logger.debug(f"Deleted job status: {job_id}")
        except Exception as e:
            logger.error(f"Error deleting job status: {str(e)}")
            raise
    
    async def get_all_job_statuses(self) -> list:
        """
        Get all job statuses (for admin/monitoring)
        Use with caution in production
        """
        try:
            keys = await self.redis.keys(f"{self.status_prefix}*")
            statuses = []
            
            for key in keys:
                status_json = await self.redis.get(key)
                if status_json:
                    statuses.append(json.loads(status_json))
            
            return statuses
            
        except Exception as e:
            logger.error(f"Error getting all job statuses: {str(e)}")
            return []
    
    async def clear_queue(self):
        """
        Clear the entire training queue
        USE WITH CAUTION - only for admin/emergency
        """
        try:
            await self.redis.delete(self.queue_name)
            logger.warning("Training queue cleared!")
        except Exception as e:
            logger.error(f"Error clearing queue: {str(e)}")
            raise


# Global instance
redis_queue = RedisQueue()
