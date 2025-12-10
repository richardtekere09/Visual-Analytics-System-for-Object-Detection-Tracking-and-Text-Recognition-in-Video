from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import shutil
import uuid
from typing import Dict
import json

from .config import settings
from .services.video_processor import VideoProcessor
from .schemas.responses import (
    VideoUploadResponse,
    ProcessingResult,
    ProcessingStatus,
    ProcessingProgress,
)

# Initialize FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Visual Analytics System for Object Detection, Tracking, and Text Recognition",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
video_processor = None
processing_status: Dict[str, Dict] = {}


@app.on_event("startup")
async def startup_event():
    """Initialize video processor on startup"""
    global video_processor
    print("[API] Starting up...")
    video_processor = VideoProcessor()
    print("[API] Ready!")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "processor_ready": video_processor is not None,
        "active_jobs": len(processing_status),
    }


@app.post("/upload", response_model=VideoUploadResponse)
async def upload_video(file: UploadFile = File(...)):
    """
    Upload a video file for processing

    Args:
        file: Video file (mp4, avi, mov, etc.)

    Returns:
        Upload confirmation with video ID
    """
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    # Check file extension
    allowed_extensions = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"}
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400, detail=f"Invalid file type. Allowed: {allowed_extensions}"
        )

    # Generate unique video ID
    video_id = str(uuid.uuid4())

    # Save uploaded file
    video_path = settings.UPLOAD_DIR / f"{video_id}{file_ext}"

    try:
        with video_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    # Check file size
    file_size = video_path.stat().st_size
    max_size = settings.MAX_VIDEO_SIZE_MB * 1024 * 1024
    if file_size > max_size:
        video_path.unlink()
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Max size: {settings.MAX_VIDEO_SIZE_MB}MB",
        )

    # Extract basic metadata
    import cv2

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        video_path.unlink()
        raise HTTPException(status_code=400, detail="Invalid video file")

    from .schemas.responses import VideoMetadata

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    cap.release()

    metadata = VideoMetadata(
        filename=file.filename,
        duration=duration,
        fps=fps,
        width=width,
        height=height,
        total_frames=total_frames,
        file_size=file_size,
    )

    # Initialize processing status
    processing_status[video_id] = {
        "status": ProcessingStatus.PENDING,
        "progress": 0,
        "current_frame": 0,
        "total_frames": total_frames,
        "video_path": str(video_path),
    }

    return VideoUploadResponse(
        video_id=video_id,
        message="Video uploaded successfully. Use /process/{video_id} to start processing.",
        metadata=metadata,
    )


@app.post("/process/{video_id}")
async def process_video(video_id: str, background_tasks: BackgroundTasks):
    """
    Start processing a video

    Args:
        video_id: Video identifier from upload

    Returns:
        Processing started confirmation
    """
    if video_id not in processing_status:
        raise HTTPException(status_code=404, detail="Video not found")

    if processing_status[video_id]["status"] == ProcessingStatus.PROCESSING:
        raise HTTPException(status_code=400, detail="Video is already being processed")

    if processing_status[video_id]["status"] == ProcessingStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Video already processed")

    # Update status
    processing_status[video_id]["status"] = ProcessingStatus.PROCESSING

    # Start background processing
    background_tasks.add_task(process_video_task, video_id)

    return {
        "video_id": video_id,
        "message": "Processing started",
        "status": ProcessingStatus.PROCESSING,
    }


def progress_callback(
    video_id: str, progress: float, current_frame: int, total_frames: int
):
    """Callback for processing progress updates"""
    processing_status[video_id].update(
        {
            "progress": progress,
            "current_frame": current_frame,
            "total_frames": total_frames,
        }
    )


async def process_video_task(video_id: str):
    """Background task for video processing"""
    try:
        video_path = Path(processing_status[video_id]["video_path"])

        # Process video
        result = video_processor.process_video(
            video_path,
            video_id,
            lambda vid, prog, curr, total: progress_callback(vid, prog, curr, total),
        )

        # Save results
        result_path = settings.RESULTS_DIR / f"{video_id}.json"
        with result_path.open("w") as f:
            json.dump(result.model_dump(mode="json"), f, indent=2, default=str)

        # Export annotated video
        print(f"[API] Exporting annotated video for {video_id}...")
        output_video_path = settings.RESULTS_DIR / f"{video_id}_annotated.mp4"
        try:
            video_processor.export_video(
                video_path,
                output_video_path,
                result,
                show_detections=True,
                show_tracks=True,
                show_ocr=True,
            )
            print(f"[API] Annotated video saved to {output_video_path}")
        except Exception as e:
            print(f"[API] Warning: Failed to export annotated video: {e}")

        # Update status
        processing_status[video_id].update(
            {
                "status": ProcessingStatus.COMPLETED,
                "progress": 100,
                "result_path": str(result_path),
                "annotated_video_path": str(output_video_path)
                if output_video_path.exists()
                else None,
            }
        )

    except Exception as e:
        print(f"[API] Error processing video {video_id}: {str(e)}")
        processing_status[video_id].update(
            {"status": ProcessingStatus.FAILED, "error": str(e)}
        )


@app.get("/status/{video_id}", response_model=ProcessingProgress)
async def get_status(video_id: str):
    """
    Get processing status for a video

    Args:
        video_id: Video identifier

    Returns:
        Current processing status and progress
    """
    if video_id not in processing_status:
        raise HTTPException(status_code=404, detail="Video not found")

    status = processing_status[video_id]

    return ProcessingProgress(
        video_id=video_id,
        status=status["status"],
        progress=status.get("progress", 0),
        current_frame=status.get("current_frame", 0),
        total_frames=status.get("total_frames", 0),
        message=f"Processing frame {status.get('current_frame', 0)}/{status.get('total_frames', 0)}",
    )


@app.get("/results/{video_id}", response_model=ProcessingResult)
async def get_results(video_id: str):
    """
    Get processing results for a video

    Args:
        video_id: Video identifier

    Returns:
        Complete processing results
    """
    if video_id not in processing_status:
        raise HTTPException(status_code=404, detail="Video not found")

    status = processing_status[video_id]

    if status["status"] != ProcessingStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Video processing not completed. Status: {status['status']}",
        )

    # Load results
    result_path = Path(status["result_path"])
    if not result_path.exists():
        raise HTTPException(status_code=404, detail="Results file not found")

    with result_path.open("r") as f:
        result_data = json.load(f)

    return ProcessingResult(**result_data)


@app.get("/results/{video_id}/download")
async def download_results(video_id: str):
    """
    Download results as JSON file

    Args:
        video_id: Video identifier

    Returns:
        JSON file with results
    """
    if video_id not in processing_status:
        raise HTTPException(status_code=404, detail="Video not found")

    status = processing_status[video_id]

    if status["status"] != ProcessingStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Processing not completed")

    result_path = Path(status["result_path"])
    if not result_path.exists():
        raise HTTPException(status_code=404, detail="Results file not found")

    return FileResponse(
        result_path, media_type="application/json", filename=f"results_{video_id}.json"
    )


@app.get("/results/{video_id}/video")
async def download_annotated_video(video_id: str):
    """
    Download annotated video with bounding boxes

    Args:
        video_id: Video identifier

    Returns:
        Annotated video file
    """
    if video_id not in processing_status:
        raise HTTPException(status_code=404, detail="Video not found")

    status = processing_status[video_id]

    if status["status"] != ProcessingStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Processing not completed")

    # Check if annotated video exists
    annotated_video_path = settings.RESULTS_DIR / f"{video_id}_annotated.mp4"
    if not annotated_video_path.exists():
        raise HTTPException(status_code=404, detail="Annotated video not found")

    return FileResponse(
        annotated_video_path,
        media_type="video/mp4",
        filename=f"annotated_{video_id}.mp4",
    )


@app.delete("/video/{video_id}")
async def delete_video(video_id: str):
    """
    Delete a video and its results

    Args:
        video_id: Video identifier

    Returns:
        Deletion confirmation
    """
    if video_id not in processing_status:
        raise HTTPException(status_code=404, detail="Video not found")

    # Delete video file
    video_path = Path(processing_status[video_id]["video_path"])
    if video_path.exists():
        video_path.unlink()

    # Delete results
    result_path = settings.RESULTS_DIR / f"{video_id}.json"
    if result_path.exists():
        result_path.unlink()

    # Remove from status
    del processing_status[video_id]

    return {"message": "Video and results deleted successfully"}


@app.get("/videos")
async def list_videos():
    """
    List all videos and their status

    Returns:
        List of videos with status
    """
    videos = []
    for video_id, status in processing_status.items():
        videos.append(
            {
                "video_id": video_id,
                "status": status["status"],
                "progress": status.get("progress", 0),
            }
        )

    return {"videos": videos, "total": len(videos)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.HOST, port=settings.PORT)
