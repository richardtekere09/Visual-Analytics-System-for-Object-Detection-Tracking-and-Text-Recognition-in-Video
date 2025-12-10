import easyocr
import numpy as np
import cv2
from typing import List, Optional
from pathlib import Path

from ..schemas.responses import OCRResult, BoundingBox, Track
from ..config import settings


class TextRecognizer:
    """EasyOCR-based text recognition"""
    
    def __init__(self, languages: List[str] = None, gpu: bool = None):
        """
        Initialize OCR reader
        
        Args:
            languages: List of language codes (e.g., ['en', 'ch_sim'])
            gpu: Whether to use GPU
        """
        self.languages = languages or settings.OCR_LANGUAGES
        self.use_gpu = gpu if gpu is not None else settings.USE_GPU
        self.confidence_threshold = settings.OCR_CONFIDENCE
        
        # Initialize reader
        print(f"[OCR] Initializing EasyOCR for languages: {self.languages}")
        self.reader = easyocr.Reader(
            self.languages,
            gpu=self.use_gpu,
            verbose=False
        )
        print(f"[OCR] Initialized successfully")
    
    def recognize(
        self,
        image: np.ndarray,
        frame_number: int,
        track_id: Optional[int] = None
    ) -> List[OCRResult]:
        """
        Recognize text in image
        
        Args:
            image: Input image (BGR format)
            frame_number: Frame number
            track_id: Optional track ID if recognizing from tracked object
            
        Returns:
            List of OCR results
        """
        # Preprocess image
        processed = self._preprocess(image)
        
        # Run OCR
        results = self.reader.readtext(processed)
        
        # Parse results
        ocr_results = []
        for bbox_coords, text, confidence in results:
            if confidence >= self.confidence_threshold:
                # Convert bbox format
                bbox = self._parse_bbox(bbox_coords)
                
                ocr_result = OCRResult(
                    text=text.strip(),
                    confidence=float(confidence),
                    bbox=bbox,
                    frame_number=frame_number,
                    track_id=track_id
                )
                ocr_results.append(ocr_result)
        
        return ocr_results
    
    def recognize_from_tracks(
        self,
        frame: np.ndarray,
        tracks: List[Track],
        frame_number: int,
        filter_classes: Optional[List[str]] = None
    ) -> List[OCRResult]:
        """
        Run OCR on tracked objects
        
        Args:
            frame: Full frame image
            tracks: List of tracks
            frame_number: Frame number
            filter_classes: Only run OCR on these classes (e.g., ['car', 'truck'])
            
        Returns:
            List of OCR results with track IDs
        """
        all_results = []
        
        for track in tracks:
            # Filter by class if specified
            if filter_classes and track.class_name not in filter_classes:
                continue
            
            # Crop region
            x1 = max(0, int(track.bbox.x1))
            y1 = max(0, int(track.bbox.y1))
            x2 = min(frame.shape[1], int(track.bbox.x2))
            y2 = min(frame.shape[0], int(track.bbox.y2))
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            crop = frame[y1:y2, x1:x2]
            
            # Skip small regions
            if crop.shape[0] < 20 or crop.shape[1] < 20:
                continue
            
            # Run OCR on crop
            results = self.recognize(crop, frame_number, track.track_id)
            
            # Adjust bbox coordinates to full frame
            for result in results:
                result.bbox.x1 += x1
                result.bbox.y1 += y1
                result.bbox.x2 += x1
                result.bbox.y2 += y1
            
            all_results.extend(results)
        
        return all_results
    
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better OCR
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        
        # Adaptive thresholding for better contrast
        binary = cv2.adaptiveThreshold(
            denoised,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )
        
        # Resize if too small (OCR works better on larger text)
        height, width = binary.shape
        if height < 64 or width < 64:
            scale = max(64 / height, 64 / width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            binary = cv2.resize(binary, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        return binary
    
    @staticmethod
    def _parse_bbox(coords: List) -> BoundingBox:
        """
        Parse EasyOCR bbox format to BoundingBox
        
        Args:
            coords: List of 4 points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            
        Returns:
            BoundingBox
        """
        # Get min/max coordinates
        x_coords = [point[0] for point in coords]
        y_coords = [point[1] for point in coords]
        
        return BoundingBox(
            x1=float(min(x_coords)),
            y1=float(min(y_coords)),
            x2=float(max(x_coords)),
            y2=float(max(y_coords))
        )
    
    def visualize(self, frame: np.ndarray, ocr_results: List[OCRResult]) -> np.ndarray:
        """
        Draw OCR results on frame
        
        Args:
            frame: Input image
            ocr_results: List of OCR results
            
        Returns:
            Annotated frame
        """
        frame_copy = frame.copy()
        
        for result in ocr_results:
            # Get coordinates
            x1, y1 = int(result.bbox.x1), int(result.bbox.y1)
            x2, y2 = int(result.bbox.x2), int(result.bbox.y2)
            
            # Draw box (green for OCR)
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw text
            label = f"{result.text} ({result.confidence:.2f})"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            # Background for text
            cv2.rectangle(frame_copy, (x1, y2), (x1 + w, y2 + h + 10), (0, 255, 0), -1)
            cv2.putText(frame_copy, label, (x1, y2 + h + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return frame_copy
    
    @staticmethod
    def should_run_ocr(class_name: str) -> bool:
        """
        Determine if OCR should run for this class
        
        Args:
            class_name: Object class name
            
        Returns:
            True if OCR should run
        """
        # Classes that commonly contain text
        text_classes = {
            'car', 'truck', 'bus', 'motorcycle',  # License plates
            'stop sign', 'traffic light',  # Traffic signs
            'book', 'laptop', 'tv', 'cell phone',  # Screens/displays
            'bottle', 'cup',  # Labels
        }
        
        return class_name.lower() in text_classes
