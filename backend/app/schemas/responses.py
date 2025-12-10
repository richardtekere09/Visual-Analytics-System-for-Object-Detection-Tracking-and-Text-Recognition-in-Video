from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class ProcessingStatus(str, Enum):
    """Video processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class BoundingBox(BaseModel):
    """Bounding box coordinates"""
    x1: float
    y1: float
    x2: float
    y2: float
    
    @property
    def width(self) -> float:
        return self.x2 - self.x1
    
    @property
    def height(self) -> float:
        return self.y2 - self.y1
    
    @property
    def center(self) -> tuple:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)


class Detection(BaseModel):
    """Single object detection"""
    class_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox


class Track(BaseModel):
    """Tracked object across frames"""
    track_id: int
    class_id: int
    class_name: str
    bbox: BoundingBox
    confidence: float
    frame_number: int


class OCRResult(BaseModel):
    """OCR text extraction result"""
    text: str
    confidence: float
    bbox: BoundingBox
    frame_number: int
    track_id: Optional[int] = None


class FrameResult(BaseModel):
    """Results for a single frame"""
    frame_number: int
    timestamp: float  # in seconds
    tracks: List[Track]
    ocr_results: List[OCRResult]


class VideoMetadata(BaseModel):
    """Video file metadata"""
    filename: str
    duration: float  # in seconds
    fps: float
    width: int
    height: int
    total_frames: int
    file_size: int  # in bytes


class ProcessingResult(BaseModel):
    """Complete processing result for a video"""
    video_id: str
    metadata: VideoMetadata
    status: ProcessingStatus
    frames: List[FrameResult]
    summary: Dict[str, Any]
    created_at: datetime
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None


class VideoUploadResponse(BaseModel):
    """Response after video upload"""
    video_id: str
    message: str
    metadata: VideoMetadata


class ProcessingProgress(BaseModel):
    """Real-time processing progress"""
    video_id: str
    status: ProcessingStatus
    progress: float = Field(..., ge=0, le=100)  # percentage
    current_frame: int
    total_frames: int
    message: str


class TrackSummary(BaseModel):
    """Summary statistics for a tracked object"""
    track_id: int
    class_name: str
    first_frame: int
    last_frame: int
    total_frames: int
    avg_confidence: float
    trajectory: List[tuple]  # List of (x, y) center points


class VideoSummary(BaseModel):
    """Summary statistics for entire video"""
    total_detections: int
    unique_tracks: int
    unique_classes: Dict[str, int]  # class_name -> count
    total_ocr_extractions: int
    unique_texts: List[str]
    processing_time: float  # in seconds
    avg_fps: float
