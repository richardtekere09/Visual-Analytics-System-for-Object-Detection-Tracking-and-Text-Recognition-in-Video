from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Application
    APP_NAME: str = "Visual Analytics System"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    UPLOAD_DIR: Path = BASE_DIR / "data" / "uploads"
    RESULTS_DIR: Path = BASE_DIR / "data" / "results"
    MODELS_DIR: Path = BASE_DIR / "models"
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Model Settings
    DETECTION_MODEL: str = "yolov8n.pt"  # n=nano, s=small, m=medium, l=large
    DETECTION_CONFIDENCE: float = 0.25
    DETECTION_IOU: float = 0.45
    
    # Tracking Settings
    TRACK_MAX_AGE: int = 30
    TRACK_MIN_HITS: int = 3
    TRACK_IOU_THRESHOLD: float = 0.3
    
    # OCR Settings
    OCR_LANGUAGES: list = ["en"]
    OCR_CONFIDENCE: float = 0.5
    
    # Processing
    MAX_VIDEO_SIZE_MB: int = 500
    FRAME_SAMPLE_RATE: int = 1  # Process every Nth frame
    BATCH_SIZE: int = 8
    
    # Performance
    USE_GPU: bool = True
    NUM_WORKERS: int = 4
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Singleton instance
settings = Settings()

# Ensure directories exist
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
