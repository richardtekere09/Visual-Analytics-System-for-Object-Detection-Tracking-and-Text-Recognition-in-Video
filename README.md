# Visual Analytics System

A comprehensive system for object detection, tracking, and text recognition in videos using state-of-the-art deep learning models.

## Features

- **Object Detection**: YOLOv8-based detection with multiple model sizes
- **Multi-Object Tracking**: ByteTrack-style tracking with Kalman filtering
- **Text Recognition (OCR)**: EasyOCR for extracting text from video frames
- **REST API**: FastAPI backend for video processing
- **Interactive Dashboard**: Streamlit frontend for visualization and analytics
- **Real-time Progress**: WebSocket-style progress tracking
- **Comprehensive Analytics**: Temporal analysis, trajectory visualization, statistics

## Architecture

```
visual-analytics-system/
├── backend/              # FastAPI backend
│   ├── app/
│   │   ├── models/      # Detection, Tracking, OCR modules
│   │   ├── services/    # Video processing pipeline
│   │   ├── schemas/     # Pydantic data models
│   │   ├── config.py    # Configuration management
│   │   └── main.py      # FastAPI application
│   └── requirements.txt
├── frontend/            # Streamlit dashboard
│   ├── app.py
│   └── requirements.txt
├── data/
│   ├── uploads/         # Uploaded videos
│   └── results/         # Processing results (JSON)
└── models/              # Downloaded ML models
```

## Installation

### Prerequisites

- Python 3.8+
- FFmpeg (for video processing)
- CUDA (optional, for GPU acceleration)

### Step 1: Clone and Setup

```bash
cd visual-analytics-system
```

### Step 2: Install Backend Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### Step 3: Install Frontend Dependencies

```bash
cd ../frontend
pip install -r requirements.txt
```

### Step 4: Download Models (Optional)

YOLOv8 models will be downloaded automatically on first run. Available models:
- `yolov8n.pt` - Nano (fastest, least accurate)
- `yolov8s.pt` - Small
- `yolov8m.pt` - Medium
- `yolov8l.pt` - Large
- `yolov8x.pt` - Extra Large (slowest, most accurate)

## Configuration

Create a `.env` file in the backend directory:

```env
# Model Settings
DETECTION_MODEL=yolov8n.pt
DETECTION_CONFIDENCE=0.25
DETECTION_IOU=0.45

# Tracking Settings
TRACK_MAX_AGE=30
TRACK_MIN_HITS=3
TRACK_IOU_THRESHOLD=0.3

# OCR Settings
OCR_LANGUAGES=["en"]
OCR_CONFIDENCE=0.5

# Processing
MAX_VIDEO_SIZE_MB=500
FRAME_SAMPLE_RATE=1
BATCH_SIZE=8
USE_GPU=true
```

## Usage

### Start the Backend Server

```bash
cd backend
python -m app.main
```

The API will be available at `http://localhost:8000`

### Start the Frontend Dashboard

```bash
cd frontend
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

### API Endpoints

#### Upload Video
```bash
POST /upload
Content-Type: multipart/form-data

# Response
{
  "video_id": "uuid",
  "message": "Video uploaded successfully",
  "metadata": {...}
}
```

#### Process Video
```bash
POST /process/{video_id}

# Response
{
  "video_id": "uuid",
  "message": "Processing started",
  "status": "processing"
}
```

#### Get Status
```bash
GET /status/{video_id}

# Response
{
  "video_id": "uuid",
  "status": "processing",
  "progress": 45.5,
  "current_frame": 450,
  "total_frames": 1000
}
```

#### Get Results
```bash
GET /results/{video_id}

# Response
{
  "video_id": "uuid",
  "metadata": {...},
  "frames": [...],
  "summary": {...}
}
```

## Using the Dashboard

1. **Upload & Process**
   - Upload a video file (MP4, AVI, MOV, MKV)
   - Click "Upload and Process"
   - Monitor real-time progress
   - Wait for completion

2. **View Results**
   - Enter your video ID
   - Click "Load Results"
   - Explore:
     - Video metadata
     - Object class distribution
     - Extracted text
     - Object trajectories
     - Frame-by-frame viewer

3. **Analytics Dashboard**
   - Temporal analysis of detections
   - Activity heatmaps
   - Text extraction trends
   - Interactive visualizations

## Example: Python API Usage

```python
import requests

# Upload video
with open('video.mp4', 'rb') as f:
    files = {'file': ('video.mp4', f, 'video/mp4')}
    response = requests.post('http://localhost:8000/upload', files=files)
    video_id = response.json()['video_id']

# Start processing
requests.post(f'http://localhost:8000/process/{video_id}')

# Check status
import time
while True:
    status = requests.get(f'http://localhost:8000/status/{video_id}').json()
    print(f"Progress: {status['progress']}%")
    
    if status['status'] == 'completed':
        break
    time.sleep(2)

# Get results
results = requests.get(f'http://localhost:8000/results/{video_id}').json()
print(f"Found {results['summary']['video_summary']['unique_tracks']} unique objects")
```

## Performance Optimization

### GPU Acceleration
- Set `USE_GPU=true` in config
- Ensure CUDA is installed
- Expect 5-10x speedup

### Frame Sampling
- Set `FRAME_SAMPLE_RATE=2` to process every 2nd frame
- Reduces processing time by 50%
- Trade-off: May miss fast-moving objects

### Batch Processing
- Increase `BATCH_SIZE` for GPUs with more memory
- Default: 8 frames per batch

### Model Selection
- Use `yolov8n.pt` for fast processing
- Use `yolov8x.pt` for highest accuracy
- Balance based on your needs

## Troubleshooting

### "API is not running"
- Ensure backend server is started
- Check port 8000 is not in use
- Verify firewall settings

### "CUDA out of memory"
- Reduce `BATCH_SIZE`
- Use smaller model (yolov8n instead of yolov8x)
- Set `USE_GPU=false` to use CPU

### "Video upload failed"
- Check file size < MAX_VIDEO_SIZE_MB
- Verify video codec is supported
- Try converting to MP4 with H.264

### OCR not detecting text
- Increase `OCR_CONFIDENCE` threshold
- Ensure text is clear and readable
- Check if object class should trigger OCR

## Technical Details

### Object Detection
- **Model**: YOLOv8 (Ultralytics)
- **Framework**: PyTorch
- **Input**: Video frames (BGR)
- **Output**: Bounding boxes, class IDs, confidences

### Object Tracking
- **Algorithm**: ByteTrack with Kalman filtering
- **State**: [x, y, width, height, vx, vy, vw, vh]
- **Matching**: Hungarian algorithm with IOU metric
- **Features**: Handle occlusions, ID recovery

### Text Recognition
- **Model**: EasyOCR
- **Preprocessing**: Grayscale, denoising, adaptive thresholding
- **Languages**: Configurable (default: English)
- **Optimization**: Only runs on relevant object classes

## Future Enhancements

- [ ] Real-time video stream processing
- [ ] Multi-camera support
- [ ] Custom object class training
- [ ] Advanced text post-processing (NER, entity linking)
- [ ] 3D trajectory reconstruction
- [ ] Export to various formats (CSV, Excel, PDF reports)
- [ ] Docker containerization
- [ ] Cloud deployment support

## License

MIT License

## Citation

If you use this system in your research, please cite:

```bibtex
@software{visual_analytics_system,
  title = {Visual Analytics System for Object Detection, Tracking, and Text Recognition},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/visual-analytics-system}
}
```

## Acknowledgments

- YOLOv8 by Ultralytics
- ByteTrack algorithm
- EasyOCR by JaidedAI
- FastAPI framework
- Streamlit framework
