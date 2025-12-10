import torch
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from typing import List, Tuple
import cv2

from ..config import settings
from ..schemas.responses import Detection, BoundingBox


class ObjectDetector:
    """YOLOv8-based object detector"""
    
    def __init__(self, model_name: str = None):
        """
        Initialize object detector
        
        Args:
            model_name: YOLO model name (e.g., 'yolov8n.pt')
        """
        self.model_name = model_name or settings.DETECTION_MODEL
        self.confidence_threshold = settings.DETECTION_CONFIDENCE
        self.iou_threshold = settings.DETECTION_IOU
        self.device = 'cuda' if settings.USE_GPU and torch.cuda.is_available() else 'cpu'
        
        # Load model
        self.model = self._load_model()
        
        print(f"[Detector] Loaded {self.model_name} on {self.device}")
    
    def _load_model(self) -> YOLO:
        """Load YOLO model"""
        model_path = settings.MODELS_DIR / self.model_name
        
        # Download if not exists
        if not model_path.exists():
            print(f"[Detector] Downloading {self.model_name}...")
            model = YOLO(self.model_name)
        else:
            model = YOLO(str(model_path))
        
        # Move to device
        model.to(self.device)
        
        return model
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect objects in a single frame
        
        Args:
            frame: Input image (BGR format)
            
        Returns:
            List of Detection objects
        """
        # Run inference
        results = self.model(
            frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            verbose=False
        )[0]
        
        # Parse results
        detections = []
        boxes = results.boxes
        
        for i in range(len(boxes)):
            # Get box coordinates
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
            
            # Get class and confidence
            class_id = int(boxes.cls[i].cpu().numpy())
            confidence = float(boxes.conf[i].cpu().numpy())
            class_name = self.model.names[class_id]
            
            # Create detection object
            detection = Detection(
                class_id=class_id,
                class_name=class_name,
                confidence=confidence,
                bbox=BoundingBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2))
            )
            detections.append(detection)
        
        return detections
    
    def detect_batch(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        """
        Detect objects in a batch of frames
        
        Args:
            frames: List of input images
            
        Returns:
            List of detection lists (one per frame)
        """
        # Run batch inference
        results = self.model(
            frames,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            verbose=False
        )
        
        # Parse results for each frame
        all_detections = []
        for result in results:
            frame_detections = []
            boxes = result.boxes
            
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                class_id = int(boxes.cls[i].cpu().numpy())
                confidence = float(boxes.conf[i].cpu().numpy())
                class_name = self.model.names[class_id]
                
                detection = Detection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    bbox=BoundingBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2))
                )
                frame_detections.append(detection)
            
            all_detections.append(frame_detections)
        
        return all_detections
    
    def get_class_names(self) -> dict:
        """Get mapping of class IDs to names"""
        return self.model.names
    
    def visualize(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """
        Draw bounding boxes on frame
        
        Args:
            frame: Input image
            detections: List of detections
            
        Returns:
            Annotated frame
        """
        frame_copy = frame.copy()
        
        for det in detections:
            # Get coordinates
            x1, y1 = int(det.bbox.x1), int(det.bbox.y1)
            x2, y2 = int(det.bbox.x2), int(det.bbox.y2)
            
            # Draw box
            color = self._get_color(det.class_id)
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{det.class_name} {det.confidence:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame_copy, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
            cv2.putText(frame_copy, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame_copy
    
    @staticmethod
    def _get_color(class_id: int) -> Tuple[int, int, int]:
        """Get consistent color for class ID"""
        np.random.seed(class_id)
        return tuple(np.random.randint(0, 255, 3).tolist())
