# 🚀 Quick Start Checklist - Retrain Button Integration

## ✅ Pre-requisites
- [ ] MongoDB Atlas connection (already configured)
- [ ] Redis Cloud connection (already configured)
- [ ] Conda environment `nammu` installed
- [ ] Node.js installed (v16+)

## 📦 Installation Steps

### 1. Install Backend Dependencies
```bash
cd backend
npm install
```
**Note:** This installs the new `axios` dependency needed to call Python Backend_2.

### 2. Verify Environment Files

**Check `backend/.env` has:**
```env
FASTAPI_URL=http://localhost:8000
```

**Check `Backend_2/.env` has:**
```env
REDIS_URL=redis://...
MONGODB_URI=mongodb+srv://...
API_PORT=8000
```

## 🏃‍♂️ Running the System (4 Terminals)

### Terminal 1: Python FastAPI Server
```bash
cd "Website final/Backend_2"
conda activate nammu
python main.py
```
✅ Expected: Server running at `http://localhost:8000`

### Terminal 2: Python Worker Process
```bash
cd "Website final/Backend_2"
conda activate nammu
python worker.py
```
✅ Expected: Worker listening for jobs

### Terminal 3: Node.js Backend
```bash
cd backend
npm run dev
```
✅ Expected: Server running at `http://localhost:5000`

### Terminal 4: React Frontend
```bash
cd frontend
npm run dev
```
✅ Expected: Frontend at `http://localhost:5173`

## 🧪 Test the Integration

1. **Health Checks:**
   - Visit: `http://localhost:8000/health` (FastAPI)
   - Visit: `http://localhost:8000/docs` (API Documentation)

2. **Login to Application:**
   - Navigate to `http://localhost:5173`
   - Login with your credentials

3. **Test Retrain Button:**
   - Go to a hotel dashboard
   - Click "Retrain Model" button in navbar
   - Look for success message: "✓ Model retraining started successfully!"

4. **Monitor Logs:**
   - **Terminal 1 (FastAPI):** Should show training job queued
   - **Terminal 2 (Worker):** Should show job processing
   - **Terminal 3 (Node.js):** Should show request forwarded

## ✅ Success Criteria

- [ ] All 4 terminals show no errors
- [ ] Health endpoint returns healthy status
- [ ] Clicking "Retrain Model" shows success message
- [ ] Worker terminal shows job processing
- [ ] No CORS errors in browser console

## 🔍 Key Changes Made

1. **`backend/controllers/train.controller.ts`**
   - Now calls Python Backend_2 via axios
   - Handles connection errors gracefully

2. **`backend/package.json`**
   - Added `axios` dependency

3. **Request Flow:**
   ```
   Frontend → Node.js → FastAPI → Redis Queue → Worker → Model Training
   ```

## 📚 Documentation

- Full integration details: `RETRAIN_INTEGRATION_GUIDE.md`
- Python backend docs: `Backend_2/README.md`
- Quick setup: `Backend_2/QUICKSTART.md`

## 🆘 Troubleshooting

**Issue:** "Training service is unavailable"
- **Fix:** Start Python Backend_2 (Terminal 1)

**Issue:** Jobs not processing
- **Fix:** Start Worker process (Terminal 2)

**Issue:** Frontend errors
- **Fix:** Check all 4 services are running

## 🎉 You're Done!

The retrain button is now fully integrated with Python Backend_2!
