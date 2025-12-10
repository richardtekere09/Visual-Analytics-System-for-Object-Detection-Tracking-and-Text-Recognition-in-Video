# Project Structure

```
visual-analytics-system/
│
├── README.md                          # Comprehensive documentation
├── QUICKSTART.md                      # Quick start guide
├── .env                               # Environment configuration
├── .gitignore                         # Git ignore rules
├── test_system.py                     # System test suite
│
├── backend/                           # FastAPI Backend
│   ├── requirements.txt               # Python dependencies
│   └── app/
│       ├── __init__.py
│       ├── config.py                  # Configuration management
│       ├── main.py                    # FastAPI application & endpoints
│       │
│       ├── models/                    # ML Models
│       │   ├── __init__.py
│       │   ├── detector.py            # YOLOv8 object detection
│       │   ├── tracker.py             # ByteTrack multi-object tracking
│       │   └── ocr.py                 # EasyOCR text recognition
│       │
│       ├── services/                  # Business Logic
│       │   ├── __init__.py
│       │   └── video_processor.py     # Video processing pipeline
│       │
│       └── schemas/                   # Data Models
│           ├── __init__.py
│           └── responses.py           # Pydantic schemas
│
├── frontend/                          # Streamlit Dashboard
│   ├── requirements.txt               # Python dependencies
│   └── app.py                         # Streamlit application
│
├── data/                              # Data Storage
│   ├── uploads/                       # Uploaded videos
│   │   └── .gitkeep
│   └── results/                       # Processing results (JSON)
│       └── .gitkeep
│
└── models/                            # Downloaded ML models
    └── (yolov8*.pt files)

```

## File Descriptions

### Root Files

- **README.md**: Complete documentation including architecture, installation, usage, API reference
- **QUICKSTART.md**: Fast setup guide for getting started quickly
- **.env**: Configuration file for model settings, thresholds, and performance tuning
- **test_system.py**: Comprehensive test suite to verify all components

### Backend (`/backend`)

#### Core Application
- **app/main.py**: FastAPI application with REST endpoints
  - `/upload` - Video upload
  - `/process/{video_id}` - Start processing
  - `/status/{video_id}` - Get progress
  - `/results/{video_id}` - Get results
  - `/videos` - List all videos

- **app/config.py**: Centralized configuration using Pydantic Settings
  - Model selection (YOLOv8 variants)
  - Detection/tracking thresholds
  - OCR settings
  - Performance parameters

#### Models (`/backend/app/models`)
- **detector.py**: YOLOv8 Object Detector
  - Single frame detection
  - Batch processing
  - Visualization
  - Class name mapping

- **tracker.py**: ByteTrack-style Tracker
  - Kalman filter state estimation
  - Hungarian algorithm matching
  - Track lifecycle management
  - ID consistency across frames

- **ocr.py**: EasyOCR Text Recognizer
  - Image preprocessing
  - Multi-language support
  - Track-based OCR
  - Confidence filtering

#### Services (`/backend/app/services`)
- **video_processor.py**: Unified Processing Pipeline
  - Frame-by-frame processing
  - Integration of detection + tracking + OCR
  - Progress callbacks
  - Summary generation
  - Annotated video export

#### Schemas (`/backend/app/schemas`)
- **responses.py**: Pydantic Data Models
  - BoundingBox, Detection, Track
  - OCRResult, FrameResult
  - VideoMetadata, ProcessingResult
  - ProcessingProgress, VideoSummary

### Frontend (`/frontend`)

- **app.py**: Streamlit Dashboard
  - **Upload & Process Page**: Video upload and processing
  - **View Results Page**: Results visualization and exploration
  - **Analytics Dashboard**: Temporal analysis and statistics

### Data (`/data`)

- **uploads/**: Temporary storage for uploaded videos
- **results/**: JSON files with complete processing results

### Models (`/models`)

- Downloaded YOLO models (auto-downloaded on first use)
- EasyOCR language models (auto-downloaded on first use)

## Component Flow

```
1. User uploads video (frontend) 
   ↓
2. FastAPI receives and saves video (main.py)
   ↓
3. Processing starts (video_processor.py)
   ↓
4. For each frame:
   - Detect objects (detector.py)
   - Track objects (tracker.py)
   - Extract text from relevant objects (ocr.py)
   ↓
5. Generate summary statistics
   ↓
6. Save results as JSON
   ↓
7. User views results in dashboard (frontend)
```

## Key Design Patterns

### 1. Singleton Pattern
- **config.py**: Single Settings instance
- **main.py**: Single VideoProcessor instance

### 2. Pipeline Pattern
- **video_processor.py**: Sequential processing stages
- Frame → Detection → Tracking → OCR → Results

### 3. Repository Pattern
- Results stored as JSON files
- File-based persistence with unique video IDs

### 4. Observer Pattern
- Progress callbacks for real-time updates
- Background task processing

### 5. Factory Pattern
- Model initialization in each component
- Automatic model downloading

## Data Flow

### Input
```
Video File (MP4/AVI/MOV/MKV)
  ↓
Frames (numpy arrays, BGR format)
```

### Processing
```
Frames
  ↓ detector.py
Detections (bbox, class, confidence)
  ↓ tracker.py
Tracks (bbox, class, track_id, confidence)
  ↓ ocr.py (on relevant objects)
OCR Results (text, bbox, confidence, track_id)
```

### Output
```
JSON File:
{
  "video_id": "...",
  "metadata": {...},
  "frames": [
    {
      "frame_number": 1,
      "timestamp": 0.033,
      "tracks": [...],
      "ocr_results": [...]
    },
    ...
  ],
  "summary": {
    "video_summary": {...},
    "track_summaries": [...]
  }
}
```

## Extension Points

### Add New Detection Model
1. Create new detector class in `models/`
2. Inherit from base interface
3. Update `config.py`

### Add New Tracking Algorithm
1. Create new tracker class in `models/`
2. Implement `update()` method
3. Update `video_processor.py`

### Add New OCR Engine
1. Create new recognizer class in `models/`
2. Implement `recognize()` method
3. Update `video_processor.py`

### Add New Visualization
1. Update `frontend/app.py`
2. Add new page or chart type
3. Use Plotly/Streamlit components

## Dependencies

### Core ML/CV
- PyTorch + Torchvision
- Ultralytics (YOLOv8)
- OpenCV
- EasyOCR
- FilterPy (Kalman filtering)
- SciPy (Hungarian algorithm)

### API/Web
- FastAPI
- Uvicorn
- Streamlit
- Requests

### Data Processing
- NumPy
- Pandas
- Pydantic

### Visualization
- Plotly
- Matplotlib
- Seaborn
