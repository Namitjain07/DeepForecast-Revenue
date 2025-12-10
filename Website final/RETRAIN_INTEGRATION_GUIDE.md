# Retrain Button Integration Guide

## 🎯 Overview

The retrain button has been successfully linked to the Python Backend_2 (FastAPI). When a user clicks the retrain button, the following flow occurs:

```
Frontend (React) 
    → Node.js Backend (Express) 
        → Python Backend_2 (FastAPI) 
            → Redis Queue 
                → Worker Process 
                    → Model Training
```

## 🔄 Request Flow

### 1. Frontend Component
**File:** `frontend/src/components/dashboard/UserNavbar.tsx`

The retrain button calls `handleRetrainModel()` which dispatches:
```typescript
dispatch(addLastTrainRecord(user.id, hotelId))
```

### 2. Frontend API Service
**File:** `frontend/src/redux/services/modelTrainAPI.ts`

Makes POST request to Node.js backend:
```typescript
POST http://localhost:5000/api/v1/train/start
Body: { userId, hotelId }
```

### 3. Node.js Backend Controller (UPDATED)
**File:** `backend/controllers/train.controller.ts`

Now forwards the request to Python Backend_2:
```typescript
POST http://localhost:8000/api/v1/train/start
Body: { userId, hotelId }
```

### 4. Python Backend_2 (FastAPI)
**File:** `Backend_2/routes.py`

- Creates a LastTrain record in MongoDB
- Enqueues training job in Redis
- Returns job details

### 5. Worker Process
**File:** `Backend_2/worker.py`

- Monitors Redis queue
- Processes training jobs asynchronously
- Updates LastTrain status in MongoDB

## 📝 Changes Made

### 1. Updated `backend/controllers/train.controller.ts`

**Before:**
- Created LastTrain record locally
- Had a TODO comment to call another API

**After:**
- Imports axios for HTTP requests
- Calls Python FastAPI Backend_2
- Handles connection errors gracefully
- Returns FastAPI response directly to frontend

### 2. Added axios Dependency

**File:** `backend/package.json`

Added `"axios": "^1.7.9"` to dependencies.

**Installation Required:**
```bash
cd backend
npm install
```

## 🚀 How to Run the Complete System

### Step 1: Start MongoDB & Redis
Ensure your MongoDB and Redis instances are running (already configured in .env files).

### Step 2: Start Python Backend_2 (FastAPI)

```bash
cd "Website final/Backend_2"

# Activate conda environment
conda activate nammu

# Start FastAPI server
python main.py
```

Server runs at: `http://localhost:8000`

### Step 3: Start Background Worker

**Open a NEW terminal:**
```bash
cd "Website final/Backend_2"

# Activate conda environment
conda activate nammu

# Start worker
python worker.py
```

Or use the PowerShell script:
```bash
.\run_worker.ps1
```

### Step 4: Start Node.js Backend

**Open a NEW terminal:**
```bash
cd backend

# Install dependencies (first time only)
npm install

# Start backend server
npm run dev
```

Server runs at: `http://localhost:5000`

### Step 5: Start Frontend

**Open a NEW terminal:**
```bash
cd frontend

# Install dependencies (first time only)
npm install

# Start frontend
npm run dev
```

Frontend runs at: `http://localhost:5173`

## 🧪 Testing the Integration

### 1. Check Health Endpoints

**FastAPI Health Check:**
```bash
curl http://localhost:8000/health
```

**Node.js Backend:**
```bash
curl http://localhost:5000/api/v1/health
```

### 2. Test Training Request

**Via Node.js Backend:**
```bash
curl -X POST http://localhost:5000/api/v1/train/start \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -d '{"userId": "USER_ID", "hotelId": "HOTEL_ID"}'
```

**Expected Response:**
```json
{
  "message": "Training job queued successfully",
  "job_id": "uuid-here",
  "train_id": "mongodb-id-here",
  "lastTrain": {
    "_id": "...",
    "hotelId": "...",
    "userId": "...",
    "status": "queued",
    "startDateTime": "...",
    "endDateTime": "..."
  }
}
```

### 3. Check Job Status

```bash
curl http://localhost:8000/api/v1/train/status/JOB_ID
```

### 4. Test from Frontend

1. Login to the application
2. Navigate to a hotel dashboard
3. Click the "Retrain Model" button in the navbar
4. Check for success message
5. Monitor the console logs in all terminals

## 📊 Monitoring & Debugging

### Check FastAPI Logs
Look for:
- `📝 Received training request for hotel: HOTEL_ID`
- `✅ Training job queued: JOB_ID`

### Check Worker Logs
Look for:
- `🎯 Processing training job: JOB_ID`
- `📊 Training models for hotel: HOTEL_ID`
- `✅ Training completed successfully`

### Check Node.js Backend Logs
Look for:
- `🚀 Starting training request for hotel: HOTEL_ID`
- `✅ FastAPI training job queued: {...}`

### Check Frontend Console
Look for:
- Training request sent
- Success/error messages
- Redux state updates

## 🔧 Environment Variables

### Node.js Backend `.env`
```env
PORT=5000
MONGODB_URI=mongodb+srv://...
JWT_SECRET=...
FASTAPI_URL=http://localhost:8000
```

### Python Backend_2 `.env`
```env
REDIS_URL=redis://...
MONGODB_URI=mongodb+srv://...
MONGODB_NAME=test
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=True
CORS_ORIGINS=http://localhost:3000,http://localhost:5173
LOG_LEVEL=INFO
```

## ⚠️ Common Issues & Solutions

### Issue 1: "Training service is unavailable"

**Cause:** Python Backend_2 is not running

**Solution:**
```bash
cd Backend_2
python main.py
```

### Issue 2: Jobs not processing

**Cause:** Worker process is not running

**Solution:**
```bash
cd Backend_2
python worker.py
```

### Issue 3: Redis connection error

**Cause:** Invalid Redis URL or Redis service is down

**Solution:** Check `Backend_2/.env` and verify Redis credentials

### Issue 4: MongoDB connection error

**Cause:** Invalid MongoDB URI or network issues

**Solution:** Check `.env` files and verify MongoDB Atlas connection

### Issue 5: CORS errors in browser

**Cause:** Frontend URL not in CORS_ORIGINS

**Solution:** Add frontend URL to `Backend_2/.env`:
```env
CORS_ORIGINS=http://localhost:3000,http://localhost:5173
```

## 📚 API Documentation

### Interactive API Docs
Visit: `http://localhost:8000/docs`

This provides:
- All available endpoints
- Request/response schemas
- Interactive testing interface

## 🎉 Success Indicators

When everything is working correctly:

1. ✅ All 4 services are running (MongoDB, Redis, FastAPI, Node.js, Frontend)
2. ✅ Health checks return "healthy" status
3. ✅ Clicking "Retrain Model" shows success message
4. ✅ Worker logs show job processing
5. ✅ LastTrain records appear in MongoDB
6. ✅ Training completes and status updates to "success"

## 📞 Need Help?

Check these resources:
- `Backend_2/README.md` - Detailed Python backend documentation
- `Backend_2/QUICKSTART.md` - Quick setup guide
- `http://localhost:8000/docs` - Interactive API documentation
