# FastAPI Model Controller (Backend_2)

## 🚀 Overview

This is the FastAPI-based model controller for DeepForecast-Revenue. It handles asynchronous model training requests using Redis queue and MongoDB for data storage.

### Architecture

```
┌─────────────────┐
│  MERN Backend   │ (Node.js/Express)
│  (Port 5000)    │
└────────┬────────┘
         │
         │ HTTP Request
         ▼
┌─────────────────┐
│  FastAPI        │ (Python)
│  (Port 8000)    │
└────────┬────────┘
         │
         ├──────────────┐
         │              │
         ▼              ▼
    ┌─────────┐   ┌──────────┐
    │  Redis  │   │ MongoDB  │
    │  Queue  │   │ Database │
    └─────────┘   └──────────┘
         │
         ▼
    ┌─────────────┐
    │   Worker    │ (Background Process)
    │   Process   │
    └─────────────┘
```

## 📁 Files Structure

```
Backend_2/
├── main.py              # FastAPI application entry point
├── routes.py            # API route definitions
├── models.py            # Pydantic models for request/response
├── database.py          # MongoDB connection manager (Motor)
├── redis_queue.py       # Redis queue manager
├── worker.py            # Background worker for processing jobs
├── predict.py           # Model training logic (NeuralProphet)
├── requirements.txt     # Python dependencies
├── .env                 # Environment variables (not in repo)
└── README.md           # This file
```

## 🔧 Setup Instructions

### 1. Prerequisites

- Python 3.8 or higher
- MongoDB instance (with connection URI)
- Redis instance (with connection URL)
- Conda/venv for environment management

### 2. Install Dependencies

```bash
# Navigate to Backend_2 directory
cd "Website final/Backend_2"

# Install Python packages
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Ensure your `.env` file contains:

```env
# MongoDB Configuration
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/
MONGODB_NAME=test

# Redis Configuration
REDIS_URL=redis://default:password@host:port

# FastAPI Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=True

# CORS Origins (comma separated)
CORS_ORIGINS=http://localhost:3000,http://localhost:5173

# Log Level
LOG_LEVEL=INFO
```

## 🏃 Running the Application

### Start the FastAPI Server

```bash
python main.py
```

Or using uvicorn directly:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Start the Background Worker

In a separate terminal:

```bash
python worker.py
```

**Important:** The worker must be running to process training jobs from the queue.

## 📡 API Endpoints

### Base URL: `http://localhost:8000/api/v1`

### 1. Start Training Job

**POST** `/train/start`

Start a new model training job for a hotel.

**Request Body:**
```json
{
  "userId": "507f1f77bcf86cd799439011",
  "hotelId": "507f191e810c19729de860ea"
}
```

**Response:**
```json
{
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
```

### 2. Get Job Status

**GET** `/train/status/{job_id}`

Get the current status of a training job.

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "train_id": "507f1f77bcf86cd799439011",
  "hotel_id": "507f191e810c19729de860ea",
  "user_id": "507f1f77bcf86cd799439011",
  "status": "running"
}
```

**Status Values:**
- `queued`: Job is in the queue waiting to be processed
- `running`: Job is currently being processed
- `success`: Job completed successfully
- `failure`: Job failed with an error

### 3. Get Last Training Record

**GET** `/train/hotel/{hotel_id}`

Get the most recent training record for a hotel.

**Response:**
```json
{
  "message": "Last train record retrieved successfully",
  "lastTrain": {
    "_id": "507f1f77bcf86cd799439011",
    "hotelId": "507f191e810c19729de860ea",
    "userId": "507f1f77bcf86cd799439011",
    "startDateTime": "2024-01-01T12:00:00",
    "endDateTime": "2024-01-01T13:30:00",
    "status": "success"
  }
}
```

### 4. Get Queue Statistics

**GET** `/train/queue/stats`

Get statistics about the training job queue.

**Response:**
```json
{
  "queue_length": 2,
  "active_jobs": 1,
  "completed_jobs": 15,
  "failed_jobs": 2
}
```

### 5. Health Check

**GET** `/health`

Check the health status of the service and its connections.

**Response:**
```json
{
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
```

## 🔄 How It Works

### Training Job Flow

1. **Client Request**: MERN backend or client sends POST request to `/api/v1/train/start`
2. **Queue Job**: FastAPI creates a `LastTrain` record in MongoDB and enqueues the job in Redis
3. **Return Response**: FastAPI immediately returns job details to client
4. **Worker Picks Up**: Background worker dequeues the job from Redis
5. **Update Status**: Worker updates job status to "running" in both Redis and MongoDB
6. **Train Models**: Worker executes the model training (NeuralProphet on 5 targets)
7. **Save Results**: Worker saves forecasts to MongoDB (optional, based on implementation)
8. **Final Update**: Worker updates job status to "success" or "failure"

### Data Flow

1. **Load Data**: `predict.py` fetches records from MongoDB `records` collection
2. **Prepare Data**: Converts records to pandas DataFrame with proper date indexing
3. **Train Models**: Trains 5 NeuralProphet models for different targets:
   - Room Revenue
   - Arrival Rooms
   - Departure Rooms
   - Compliment Rooms
   - House Use Rooms
4. **Return Forecasts**: Returns forecast data as JSON-serializable dictionaries

## 🧪 Testing

### Test with cURL

```bash
# Start training
curl -X POST http://localhost:8000/api/v1/train/start \
  -H "Content-Type: application/json" \
  -d '{"userId": "507f1f77bcf86cd799439011", "hotelId": "507f191e810c19729de860ea"}'

# Check job status
curl http://localhost:8000/api/v1/train/status/{job_id}

# Check last train
curl http://localhost:8000/api/v1/train/hotel/{hotel_id}

# Health check
curl http://localhost:8000/health
```

### Test with Python

```python
import requests

# Start training
response = requests.post(
    "http://localhost:8000/api/v1/train/start",
    json={
        "userId": "507f1f77bcf86cd799439011",
        "hotelId": "507f191e810c19729de860ea"
    }
)
print(response.json())
```

## 📊 MongoDB Collections Used

### 1. `records`
Stores historical hotel data (input for training)

### 2. `lasttrains`
Stores training job metadata and status

### 3. `forecasts`
Stores model predictions (to be implemented in worker)

## 🔑 Key Features

- ✅ **Asynchronous Processing**: Non-blocking API using Redis queue
- ✅ **Real-time Status**: Track job status in real-time
- ✅ **MongoDB Integration**: Fetches data directly from MongoDB
- ✅ **Redis Queue**: Reliable job queue with blocking operations
- ✅ **Background Worker**: Separate process for CPU-intensive tasks
- ✅ **Health Monitoring**: Health check endpoint for monitoring
- ✅ **Structured Logging**: Clear, colorized logs with loguru
- ✅ **CORS Support**: Configurable CORS for frontend integration

## 🛠️ Development Tips

### Running in Development Mode

```bash
# Terminal 1: Start FastAPI with auto-reload
python main.py

# Terminal 2: Start worker
python worker.py
```

### Monitoring Logs

Both the FastAPI server and worker output detailed logs:
- 🚀 Startup/shutdown events
- 📥 Job enqueue/dequeue
- 🤖 Model training progress
- ✅ Success/failure status

### Debugging

Enable debug logging by setting in `.env`:
```env
LOG_LEVEL=DEBUG
```

## 🔗 Integration with MERN Backend

The MERN backend can call this FastAPI service to trigger training:

```typescript
// In Node.js/Express backend
const response = await axios.post('http://localhost:8000/api/v1/train/start', {
  userId: req.body.userId,
  hotelId: req.body.hotelId
});
```

## 🐛 Troubleshooting

### Redis Connection Issues
- Verify Redis is running and accessible
- Check REDIS_URL in .env file
- Test connection: `redis-cli ping`

### MongoDB Connection Issues
- Verify MongoDB URI is correct
- Check network connectivity
- Ensure database name exists

### Worker Not Processing Jobs
- Ensure worker.py is running in a separate terminal
- Check worker logs for errors
- Verify Redis queue has jobs: Check `/train/queue/stats`

## 📝 Notes

- The worker processes one job at a time (can be scaled to multiple workers)
- Job statuses in Redis expire after 24 hours
- Model training can take several minutes depending on data size
- Ensure sufficient system resources (CPU/RAM) for training

## 🎯 Future Enhancements

- [ ] Save forecast results to MongoDB `forecasts` collection
- [ ] Support for multiple concurrent workers
- [ ] Job prioritization in queue
- [ ] Webhook notifications on job completion
- [ ] Admin dashboard for queue management
- [ ] Model versioning and rollback
- [ ] Automated testing suite
- [ ] Docker containerization
- [ ] Kubernetes deployment manifests

## 📄 License

Part of the DeepForecast-Revenue project.

---

**Maintainer:** DeepForecast Team  
**Last Updated:** December 2024
