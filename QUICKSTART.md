# Quick Start Guide

## Installation (5 minutes)

### 1. Install Backend Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 2. Install Frontend Dependencies
```bash
cd ../frontend
pip install -r requirements.txt
```

### 3. Run Tests
```bash
cd ..
python test_system.py
```

## Running the System (2 terminals needed)

### Terminal 1: Start Backend
```bash
cd backend
python -m app.main
```

Expected output:
```
[Detector] Loaded yolov8n.pt on cuda
[Tracker] Initialized (max_age=30, min_hits=3)
[OCR] Initializing EasyOCR for languages: ['en']
[VideoProcessor] Ready!
[API] Ready!
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Terminal 2: Start Frontend
```bash
cd frontend
streamlit run app.py
```

Browser will automatically open to `http://localhost:8501`

## First Video Processing

1. **Open Dashboard** (http://localhost:8501)

2. **Upload Video**
   - Click "Upload & Process"
   - Choose a video file (MP4, AVI, MOV, MKV)
   - Click "Upload and Process"

3. **Monitor Progress**
   - Watch real-time progress bar
   - Wait for "Processing completed!"

4. **View Results**
   - Go to "View Results" page
   - Enter your video ID
   - Click "Load Results"
   - Explore visualizations

## Testing with Sample Video

Create a test video:
```bash
python test_system.py
```

This creates `data/uploads/test_video.mp4` (3 seconds, moving rectangle with text)

## Common Issues

### "ModuleNotFoundError"
```bash
pip install -r backend/requirements.txt
pip install -r frontend/requirements.txt
```

### "CUDA out of memory"
Edit `.env`:
```
USE_GPU=false
BATCH_SIZE=4
```

### "Port already in use"
Kill existing processes:
```bash
# Linux/Mac
lsof -ti:8000 | xargs kill -9
lsof -ti:8501 | xargs kill -9

# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

## Performance Tips

### Fast Processing (Lower Accuracy)
```
DETECTION_MODEL=yolov8n.pt
FRAME_SAMPLE_RATE=2
```

### High Accuracy (Slower)
```
DETECTION_MODEL=yolov8x.pt
FRAME_SAMPLE_RATE=1
```

### GPU Acceleration
```
USE_GPU=true
BATCH_SIZE=16
```

## Next Steps

- Read full [README.md](README.md)
- Try different YOLO models
- Adjust confidence thresholds
- Process your own videos
- Explore analytics dashboard

## Support

For issues, check:
1. Backend logs in terminal 1
2. Frontend logs in terminal 2
3. Browser console (F12)

Happy analyzing! 🎉
