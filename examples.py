#!/usr/bin/env python3
"""
Example: Using Visual Analytics System Programmatically

This script demonstrates how to use the system without the web interface.
"""

import sys
from pathlib import Path
import cv2
import numpy as np

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from app.models.detector import ObjectDetector
from app.models.tracker import ObjectTracker
from app.models.ocr import TextRecognizer
from app.services.video_processor import VideoProcessor


def example_1_process_single_frame():
    """Example 1: Process a single frame"""
    print("="*60)
    print("Example 1: Process Single Frame")
    print("="*60)
    
    # Create a test image
    image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    # Draw some objects
    cv2.rectangle(image, (100, 100), (200, 200), (0, 255, 0), -1)
    cv2.putText(image, "CAR123", (110, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Initialize components
    detector = ObjectDetector()
    
    # Detect objects
    detections = detector.detect(image)
    
    print(f"\nFound {len(detections)} objects:")
    for det in detections:
        print(f"  - {det.class_name}: {det.confidence:.2f} at ({det.bbox.x1:.0f}, {det.bbox.y1:.0f})")
    
    # Visualize
    vis_image = detector.visualize(image, detections)
    
    # Save result
    output_path = Path("data/results/example1_result.jpg")
    cv2.imwrite(str(output_path), vis_image)
    print(f"\nSaved visualization to: {output_path}")


def example_2_track_objects():
    """Example 2: Track objects across multiple frames"""
    print("\n" + "="*60)
    print("Example 2: Track Objects Across Frames")
    print("="*60)
    
    # Initialize
    detector = ObjectDetector()
    tracker = ObjectTracker()
    
    # Simulate 10 frames with a moving object
    for frame_num in range(1, 11):
        # Create frame
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Draw moving rectangle
        x = 50 + frame_num * 50
        cv2.rectangle(image, (x, 200), (x + 100, 300), (0, 255, 0), -1)
        
        # Detect
        detections = detector.detect(image)
        
        # Track
        tracks = tracker.update(detections, frame_num)
        
        print(f"\nFrame {frame_num}:")
        for track in tracks:
            print(f"  Track {track.track_id}: {track.class_name} at ({track.bbox.x1:.0f}, {track.bbox.y1:.0f})")
        
        # Visualize last frame
        if frame_num == 10:
            vis_image = tracker.visualize(image, tracks)
            output_path = Path("data/results/example2_result.jpg")
            cv2.imwrite(str(output_path), vis_image)
            print(f"\nSaved last frame to: {output_path}")


def example_3_extract_text():
    """Example 3: Extract text from image"""
    print("\n" + "="*60)
    print("Example 3: Extract Text with OCR")
    print("="*60)
    
    # Create image with text
    image = np.ones((200, 600, 3), dtype=np.uint8) * 255
    cv2.putText(image, "LICENSE PLATE", (50, 80), 
               cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    cv2.putText(image, "ABC-1234", (100, 150), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    
    # Initialize OCR
    print("\nInitializing OCR (this may take a moment)...")
    ocr = TextRecognizer()
    
    # Recognize text
    results = ocr.recognize(image, frame_number=1)
    
    print(f"\nFound {len(results)} text regions:")
    for result in results:
        print(f"  - '{result.text}' (confidence: {result.confidence:.2f})")
    
    # Visualize
    vis_image = ocr.visualize(image, results)
    output_path = Path("data/results/example3_result.jpg")
    cv2.imwrite(str(output_path), vis_image)
    print(f"\nSaved visualization to: {output_path}")


def example_4_process_video():
    """Example 4: Process complete video"""
    print("\n" + "="*60)
    print("Example 4: Process Complete Video")
    print("="*60)
    
    # Create test video if it doesn't exist
    video_path = Path("data/uploads/test_video.mp4")
    
    if not video_path.exists():
        print("\nCreating test video...")
        create_test_video(video_path)
    
    print(f"\nProcessing video: {video_path}")
    
    # Initialize processor
    processor = VideoProcessor()
    
    # Process video
    result = processor.process_video(
        video_path,
        video_id="example4",
        progress_callback=lambda vid, prog, curr, total: 
            print(f"  Progress: {prog:.1f}% ({curr}/{total} frames)", end='\r')
    )
    
    print("\n\nProcessing complete!")
    print(f"  Total detections: {result.summary['video_summary']['total_detections']}")
    print(f"  Unique tracks: {result.summary['video_summary']['unique_tracks']}")
    print(f"  OCR extractions: {result.summary['video_summary']['total_ocr_extractions']}")
    print(f"  Processing time: {result.summary['video_summary']['processing_time']:.2f}s")
    
    # Export annotated video
    output_path = Path("data/results/example4_annotated.mp4")
    print(f"\nExporting annotated video to: {output_path}")
    processor.export_video(video_path, output_path, result)
    print("Done!")


def create_test_video(output_path: Path):
    """Create a test video with moving objects and text"""
    # Video parameters
    width, height = 640, 480
    fps = 30
    duration = 3  # seconds
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    # Generate frames
    for i in range(fps * duration):
        frame = np.random.randint(0, 50, (height, width, 3), dtype=np.uint8)
        
        # Draw moving rectangle (simulated vehicle)
        x = int((width - 150) * (i / (fps * duration)))
        cv2.rectangle(frame, (x, 150), (x + 150, 250), (0, 255, 0), -1)
        
        # Add "license plate"
        cv2.rectangle(frame, (x + 40, 220), (x + 110, 240), (255, 255, 255), -1)
        cv2.putText(frame, "ABC123", (x + 45, 235),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Add frame counter
        cv2.putText(frame, f"Frame {i+1}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    print(f"  Created test video: {output_path}")


def example_5_custom_pipeline():
    """Example 5: Custom processing pipeline"""
    print("\n" + "="*60)
    print("Example 5: Custom Processing Pipeline")
    print("="*60)
    
    # Initialize components with custom settings
    from app.config import settings
    
    detector = ObjectDetector(model_name="yolov8n.pt")
    tracker = ObjectTracker(max_age=50, min_hits=5)
    
    print("\nCustom pipeline initialized:")
    print(f"  Detector model: yolov8n.pt")
    print(f"  Tracker max_age: 50 frames")
    print(f"  Tracker min_hits: 5 frames")
    
    # Process a few frames
    for frame_num in range(1, 6):
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Only detect objects with high confidence
        detections = detector.detect(image)
        high_conf_detections = [d for d in detections if d.confidence > 0.5]
        
        # Track only high-confidence detections
        tracks = tracker.update(high_conf_detections, frame_num)
        
        print(f"\nFrame {frame_num}: {len(detections)} detections → {len(high_conf_detections)} high-conf → {len(tracks)} tracks")


def main():
    """Run all examples"""
    print("\n" + "="*60)
    print("Visual Analytics System - Usage Examples")
    print("="*60 + "\n")
    
    examples = [
        ("Single Frame Detection", example_1_process_single_frame),
        ("Multi-Frame Tracking", example_2_track_objects),
        ("Text Recognition (OCR)", example_3_extract_text),
        ("Complete Video Processing", example_4_process_video),
        ("Custom Pipeline", example_5_custom_pipeline),
    ]
    
    print("Available examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    
    print("\nRunning all examples...\n")
    
    for name, func in examples:
        try:
            func()
        except Exception as e:
            print(f"\n❌ {name} failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("All examples completed!")
    print("Check data/results/ for output files")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
