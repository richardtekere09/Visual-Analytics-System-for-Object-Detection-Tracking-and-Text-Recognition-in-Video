import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
import time
from datetime import datetime
import json

from ..models.detector import ObjectDetector
from ..models.tracker import ObjectTracker
from ..models.ocr import TextRecognizer
from ..schemas.responses import (
    FrameResult,
    VideoMetadata,
    ProcessingResult,
    ProcessingStatus,
    VideoSummary,
    TrackSummary,
    Track,
)
from ..config import settings


class VideoProcessor:
    """Unified video processing pipeline"""

    def __init__(self):
        """Initialize all components"""
        print("[VideoProcessor] Initializing components...")

        self.detector = ObjectDetector()
        self.tracker = ObjectTracker()

        # Try to initialize OCR, but make it optional
        try:
            self.ocr = TextRecognizer()
            self.ocr_available = True
            print("[VideoProcessor] OCR initialized successfully")
        except Exception as e:
            print(f"[VideoProcessor] Warning: OCR initialization failed: {e}")
            print("[VideoProcessor] Continuing without OCR support")
            self.ocr = None
            self.ocr_available = False

        self.frame_sample_rate = settings.FRAME_SAMPLE_RATE
        self.batch_size = settings.BATCH_SIZE

        print("[VideoProcessor] Ready!")

    def process_video(
        self,
        video_path: Path,
        video_id: str,
        progress_callback: Optional[Callable] = None,
    ) -> ProcessingResult:
        """
        Process entire video

        Args:
            video_path: Path to video file
            video_id: Unique video identifier
            progress_callback: Optional callback for progress updates

        Returns:
            ProcessingResult with all detections, tracks, and OCR
        """
        start_time = time.time()

        try:
            # Open video
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {video_path}")

            # Get metadata
            metadata = self._extract_metadata(cap, video_path)

            # Reset tracker
            self.tracker.reset()

            # Process frames
            frame_results = []
            frame_number = 0
            processed_frames = 0

            print(f"[VideoProcessor] Processing {metadata.total_frames} frames...")

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_number += 1

                # Sample frames
                if frame_number % self.frame_sample_rate != 0:
                    continue

                try:
                    # Process frame
                    frame_result = self._process_frame(
                        frame, frame_number, metadata.fps
                    )
                    frame_results.append(frame_result)

                    processed_frames += 1

                    # Progress callback
                    if progress_callback and processed_frames % 10 == 0:
                        progress = (frame_number / metadata.total_frames) * 100
                        progress_callback(
                            video_id, progress, frame_number, metadata.total_frames
                        )
                except Exception as e:
                    print(
                        f"[VideoProcessor] Error processing frame {frame_number}: {e}"
                    )
                    # Continue with next frame instead of crashing
                    continue

            cap.release()

            # Generate summary
            summary = self._generate_summary(
                frame_results, time.time() - start_time, metadata
            )

            # Create result
            result = ProcessingResult(
                video_id=video_id,
                metadata=metadata,
                status=ProcessingStatus.COMPLETED,
                frames=frame_results,
                summary=summary,
                created_at=datetime.now(),
                completed_at=datetime.now(),
            )

            print(f"[VideoProcessor] Completed in {time.time() - start_time:.2f}s")

            return result

        except Exception as e:
            print(f"[VideoProcessor] Fatal error: {e}")
            import traceback

            traceback.print_exc()
            raise

    def _process_frame(
        self, frame: np.ndarray, frame_number: int, fps: float
    ) -> FrameResult:
        """
        Process single frame

        Args:
            frame: Video frame
            frame_number: Frame index
            fps: Video FPS

        Returns:
            FrameResult with detections, tracks, and OCR
        """
        timestamp = frame_number / fps

        # Step 1: Detect objects
        detections = self.detector.detect(frame)

        # Step 2: Track objects
        tracks = self.tracker.update(detections, frame_number)

        # Step 3: Run OCR on relevant objects (if OCR is available)
        ocr_results = []
        if self.ocr_available and self.ocr:
            for track in tracks:
                if TextRecognizer.should_run_ocr(track.class_name):
                    # Crop region
                    x1 = max(0, int(track.bbox.x1))
                    y1 = max(0, int(track.bbox.y1))
                    x2 = min(frame.shape[1], int(track.bbox.x2))
                    y2 = min(frame.shape[0], int(track.bbox.y2))

                    if x2 > x1 and y2 > y1:
                        crop = frame[y1:y2, x1:x2]

                        # Skip small regions
                        if crop.shape[0] >= 20 and crop.shape[1] >= 20:
                            results = self.ocr.recognize(
                                crop, frame_number, track.track_id
                            )

                            # Adjust coordinates to full frame
                            for result in results:
                                result.bbox.x1 += x1
                                result.bbox.y1 += y1
                                result.bbox.x2 += x1
                                result.bbox.y2 += y1

                            ocr_results.extend(results)

        return FrameResult(
            frame_number=frame_number,
            timestamp=timestamp,
            tracks=tracks,
            ocr_results=ocr_results,
        )

    def _extract_metadata(
        self, cap: cv2.VideoCapture, video_path: Path
    ) -> VideoMetadata:
        """Extract video metadata"""
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        file_size = video_path.stat().st_size

        return VideoMetadata(
            filename=video_path.name,
            duration=duration,
            fps=fps,
            width=width,
            height=height,
            total_frames=total_frames,
            file_size=file_size,
        )

    def _generate_summary(
        self,
        frame_results: List[FrameResult],
        processing_time: float,
        metadata: VideoMetadata,
    ) -> Dict[str, Any]:
        """Generate summary statistics"""

        # Track statistics
        track_dict: Dict[int, List[Track]] = {}
        class_counts: Dict[str, int] = {}
        all_texts = []

        total_detections = 0

        for frame_result in frame_results:
            total_detections += len(frame_result.tracks)

            for track in frame_result.tracks:
                # Group by track ID
                if track.track_id not in track_dict:
                    track_dict[track.track_id] = []
                track_dict[track.track_id].append(track)

                # Count classes
                class_counts[track.class_name] = (
                    class_counts.get(track.class_name, 0) + 1
                )

            # Collect OCR texts
            for ocr_result in frame_result.ocr_results:
                all_texts.append(ocr_result.text)

        # Track summaries
        track_summaries = []
        for track_id, tracks in track_dict.items():
            trajectory = [track.bbox.center for track in tracks]
            avg_confidence = sum(t.confidence for t in tracks) / len(tracks)

            summary = TrackSummary(
                track_id=track_id,
                class_name=tracks[0].class_name,
                first_frame=tracks[0].frame_number,
                last_frame=tracks[-1].frame_number,
                total_frames=len(tracks),
                avg_confidence=avg_confidence,
                trajectory=trajectory,
            )
            track_summaries.append(summary)

        # Unique texts (case-insensitive)
        unique_texts = list(set(text.lower() for text in all_texts))

        # Create summary
        summary = VideoSummary(
            total_detections=total_detections,
            unique_tracks=len(track_dict),
            unique_classes=class_counts,
            total_ocr_extractions=len(all_texts),
            unique_texts=unique_texts[:50],  # Limit to top 50
            processing_time=processing_time,
            avg_fps=metadata.total_frames / processing_time
            if processing_time > 0
            else 0,
        )

        return {
            "video_summary": summary.model_dump(),
            "track_summaries": [ts.model_dump() for ts in track_summaries],
        }

    def visualize_frame(
        self,
        frame: np.ndarray,
        frame_result: FrameResult,
        show_detections: bool = True,
        show_tracks: bool = True,
        show_ocr: bool = True,
    ) -> np.ndarray:
        """
        Visualize processing results on frame with different colors for each component

        Args:
            frame: Original frame
            frame_result: Processing results
            show_detections: Show detection boxes
            show_tracks: Show track IDs
            show_ocr: Show OCR results

        Returns:
            Annotated frame
        """
        vis_frame = frame.copy()

        # Draw tracks with unique colors per track ID
        if show_tracks and frame_result.tracks:
            for track in frame_result.tracks:
                x1, y1 = int(track.bbox.x1), int(track.bbox.y1)
                x2, y2 = int(track.bbox.x2), int(track.bbox.y2)

                # Get consistent color for this track ID
                np.random.seed(track.track_id)
                color = tuple(np.random.randint(100, 255, 3).tolist())

                # Draw thick box for track
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 3)

                # Draw track ID label with background
                label = f"ID:{track.track_id} {track.class_name}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                thickness = 2
                (label_w, label_h), _ = cv2.getTextSize(
                    label, font, font_scale, thickness
                )

                # Draw label background
                cv2.rectangle(
                    vis_frame,
                    (x1, y1 - label_h - 12),
                    (x1 + label_w + 6, y1),
                    color,
                    -1,
                )

                # Draw label text
                cv2.putText(
                    vis_frame,
                    label,
                    (x1 + 3, y1 - 6),
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                )

                # Draw confidence
                conf_text = f"{track.confidence:.2f}"
                cv2.putText(vis_frame, conf_text, (x1 + 3, y2 - 6), font, 0.5, color, 1)

        # Draw OCR results with green boxes
        if show_ocr and frame_result.ocr_results:
            for ocr_result in frame_result.ocr_results:
                x1, y1 = int(ocr_result.bbox.x1), int(ocr_result.bbox.y1)
                x2, y2 = int(ocr_result.bbox.x2), int(ocr_result.bbox.y2)

                # Green color for OCR
                ocr_color = (0, 255, 0)

                # Draw OCR box
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), ocr_color, 2)

                # Draw text label
                text_label = f"TEXT: {ocr_result.text}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2
                (text_w, text_h), _ = cv2.getTextSize(
                    text_label, font, font_scale, thickness
                )

                # Draw text background
                cv2.rectangle(
                    vis_frame,
                    (x1, y2),
                    (x1 + text_w + 6, y2 + text_h + 10),
                    ocr_color,
                    -1,
                )

                # Draw text
                cv2.putText(
                    vis_frame,
                    text_label,
                    (x1 + 3, y2 + text_h + 5),
                    font,
                    font_scale,
                    (0, 0, 0),
                    thickness,
                )

        # Add frame info overlay
        info_bg_height = 90
        overlay = vis_frame.copy()
        cv2.rectangle(
            overlay, (0, 0), (vis_frame.shape[1], info_bg_height), (0, 0, 0), -1
        )
        cv2.addWeighted(overlay, 0.6, vis_frame, 0.4, 0, vis_frame)

        # Frame number and time
        info_text = (
            f"Frame: {frame_result.frame_number} | Time: {frame_result.timestamp:.2f}s"
        )
        cv2.putText(
            vis_frame,
            info_text,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        # Statistics
        stats_text = (
            f"Tracks: {len(frame_result.tracks)} | OCR: {len(frame_result.ocr_results)}"
        )
        cv2.putText(
            vis_frame,
            stats_text,
            (10, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        # Legend
        legend_y = 80
        if show_tracks:
            cv2.putText(
                vis_frame,
                "Colored boxes = Tracked objects",
                (10, legend_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
            )

        return vis_frame

    def export_video(
        self,
        input_path: Path,
        output_path: Path,
        result: ProcessingResult,
        show_detections: bool = True,
        show_tracks: bool = True,
        show_ocr: bool = True,
    ):
        """
        Export annotated video

        Args:
            input_path: Original video path
            output_path: Output video path
            result: Processing result
            show_detections: Show detection boxes
            show_tracks: Show track IDs
            show_ocr: Show OCR results
        """
        cap = cv2.VideoCapture(str(input_path))

        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            str(output_path),
            fourcc,
            result.metadata.fps,
            (result.metadata.width, result.metadata.height),
        )

        # Create frame lookup
        frame_lookup = {fr.frame_number: fr for fr in result.frames}

        frame_number = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_number += 1

            # Annotate if we have results for this frame
            if frame_number in frame_lookup:
                frame_result = frame_lookup[frame_number]
                frame = self.visualize_frame(
                    frame, frame_result, show_detections, show_tracks, show_ocr
                )

            out.write(frame)

        cap.release()
        out.release()

        print(f"[VideoProcessor] Exported annotated video to {output_path}")
