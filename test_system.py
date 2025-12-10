#!/usr/bin/env python3
"""
Test script for Visual Analytics System
"""

import sys
from pathlib import Path
import numpy as np
import cv2

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

def test_imports():
    """Test if all dependencies are installed"""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        import cv2
        print(f"✓ OpenCV {cv2.__version__}")
    except ImportError as e:
        print(f"✗ OpenCV import failed: {e}")
        return False
    
    try:
        from ultralytics import YOLO
        print(f"✓ Ultralytics YOLO")
    except ImportError as e:
        print(f"✗ Ultralytics import failed: {e}")
        return False
    
    try:
        import easyocr
        print(f"✓ EasyOCR")
    except ImportError as e:
        print(f"✗ EasyOCR import failed: {e}")
        return False
    
    try:
        import fastapi
        print(f"✓ FastAPI {fastapi.__version__}")
    except ImportError as e:
        print(f"✗ FastAPI import failed: {e}")
        return False
    
    try:
        import streamlit
        print(f"✓ Streamlit {streamlit.__version__}")
    except ImportError as e:
        print(f"✗ Streamlit import failed: {e}")
        return False
    
    print("\n✓ All imports successful!\n")
    return True


def test_detector():
    """Test object detector"""
    print("Testing Object Detector...")
    
    try:
        from app.models.detector import ObjectDetector
        
        # Create test image
        test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # Initialize detector
        detector = ObjectDetector()
        
        # Run detection
        detections = detector.detect(test_image)
        
        print(f"✓ Detector initialized and ran successfully")
        print(f"  Found {len(detections)} objects in test image")
        
        return True
        
    except Exception as e:
        print(f"✗ Detector test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tracker():
    """Test object tracker"""
    print("\nTesting Object Tracker...")
    
    try:
        from app.models.tracker import ObjectTracker
        from app.schemas.responses import Detection, BoundingBox
        
        # Initialize tracker
        tracker = ObjectTracker()
        
        # Create test detection
        test_detection = Detection(
            class_id=0,
            class_name="person",
            confidence=0.9,
            bbox=BoundingBox(x1=100, y1=100, x2=200, y2=200)
        )
        
        # Update tracker
        tracks = tracker.update([test_detection], frame_number=1)
        
        print(f"✓ Tracker initialized and ran successfully")
        print(f"  Created {len(tracks)} tracks")
        
        return True
        
    except Exception as e:
        print(f"✗ Tracker test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ocr():
    """Test OCR module"""
    print("\nTesting OCR Module...")
    
    try:
        from app.models.ocr import TextRecognizer
        
        # Create test image with text
        test_image = np.ones((100, 300, 3), dtype=np.uint8) * 255
        cv2.putText(test_image, "TEST123", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Initialize OCR
        print("  Initializing EasyOCR (this may take a moment)...")
        ocr = TextRecognizer()
        
        # Run OCR
        results = ocr.recognize(test_image, frame_number=1)
        
        print(f"✓ OCR initialized and ran successfully")
        print(f"  Detected {len(results)} text regions")
        
        return True
        
    except Exception as e:
        print(f"✗ OCR test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_video_processor():
    """Test video processor"""
    print("\nTesting Video Processor...")
    
    try:
        from app.services.video_processor import VideoProcessor
        
        # Initialize processor
        processor = VideoProcessor()
        
        print(f"✓ Video processor initialized successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Video processor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_api():
    """Test API endpoints"""
    print("\nTesting API...")
    
    try:
        from app.main import app
        from fastapi.testclient import TestClient
        
        client = TestClient(app)
        
        # Test health endpoint
        response = client.get("/health")
        assert response.status_code == 200
        
        print(f"✓ API initialized successfully")
        print(f"  Health check: {response.json()}")
        
        return True
        
    except Exception as e:
        print(f"✗ API test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_test_video():
    """Create a simple test video"""
    print("\nCreating test video...")
    
    try:
        output_path = Path("data/uploads/test_video.mp4")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Video parameters
        width, height = 640, 480
        fps = 30
        duration = 3  # seconds
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # Generate frames
        for i in range(fps * duration):
            frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            
            # Draw moving rectangle
            x = int((width - 100) * (i / (fps * duration)))
            cv2.rectangle(frame, (x, 100), (x + 100, 200), (0, 255, 0), -1)
            
            # Add text
            cv2.putText(frame, f"Frame {i}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            out.write(frame)
        
        out.release()
        
        print(f"✓ Test video created: {output_path}")
        print(f"  {width}x{height}, {fps} fps, {duration}s")
        
        return True
        
    except Exception as e:
        print(f"✗ Video creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("="*60)
    print("Visual Analytics System - Test Suite")
    print("="*60 + "\n")
    
    tests = [
        ("Imports", test_imports),
        ("Detector", test_detector),
        ("Tracker", test_tracker),
        ("OCR", test_ocr),
        ("Video Processor", test_video_processor),
        ("API", test_api),
        ("Test Video", create_test_video),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n✗ {test_name} test crashed: {e}")
            results[test_name] = False
        
        print()
    
    # Summary
    print("="*60)
    print("Test Summary")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status} - {test_name}")
    
    total = len(results)
    passed = sum(results.values())
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! System is ready to use.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
