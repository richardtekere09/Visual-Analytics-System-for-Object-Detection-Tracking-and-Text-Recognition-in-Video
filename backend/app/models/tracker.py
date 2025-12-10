import numpy as np
from typing import List, Tuple
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
import cv2

from ..schemas.responses import Detection, Track, BoundingBox
from ..config import settings


class KalmanBoxTracker:
    """Kalman Filter for tracking bounding boxes"""

    count = 0  # Global track ID counter

    def __init__(self, bbox: BoundingBox, class_id: int, class_name: str):
        """Initialize Kalman filter for bbox tracking"""
        # State: [x, y, w, h, vx, vy, vw, vh]
        self.kf = KalmanFilter(dim_x=8, dim_z=4)

        # State transition matrix
        self.kf.F = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0, 0],
                [0, 1, 0, 0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 0, 1, 0],
                [0, 0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
            ]
        )

        # Measurement matrix
        self.kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
            ]
        )

        # Covariance matrices
        self.kf.R *= 10.0  # Measurement uncertainty
        self.kf.P *= 10.0  # Initial uncertainty
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        # Initialize state
        x, y, w, h = self._bbox_to_xywh(bbox)
        self.kf.x[:4] = [[x], [y], [w], [h]]  # Fix: Use column vector format

        # Track properties
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.class_id = class_id
        self.class_name = class_name
        self.hits = 1
        self.age = 0
        self.time_since_update = 0
        self.history = []

    def update(self, bbox: BoundingBox, class_id: int, class_name: str):
        """Update tracker with new detection"""
        self.time_since_update = 0
        self.hits += 1
        self.class_id = class_id
        self.class_name = class_name

        x, y, w, h = self._bbox_to_xywh(bbox)
        self.kf.update([[x], [y], [w], [h]])  # Fix: Use column vector format

    def predict(self) -> BoundingBox:
        """Predict next state and return bbox"""
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1

        state = self.kf.x[:4].flatten()
        bbox = self._xywh_to_bbox(state)
        self.history.append(bbox)

        return bbox

    def get_state(self) -> BoundingBox:
        """Get current bbox state"""
        state = self.kf.x[:4].flatten()
        return self._xywh_to_bbox(state)

    @staticmethod
    def _bbox_to_xywh(bbox: BoundingBox) -> np.ndarray:
        """Convert bbox to [x_center, y_center, width, height]"""
        w = bbox.x2 - bbox.x1
        h = bbox.y2 - bbox.y1
        x = bbox.x1 + w / 2
        y = bbox.y1 + h / 2
        return np.array([x, y, w, h])

    @staticmethod
    def _xywh_to_bbox(xywh: np.ndarray) -> BoundingBox:
        """Convert [x_center, y_center, width, height] to bbox"""
        x, y, w, h = xywh
        return BoundingBox(
            x1=float(x - w / 2),
            y1=float(y - h / 2),
            x2=float(x + w / 2),
            y2=float(y + h / 2),
        )


class ObjectTracker:
    """ByteTrack-style multi-object tracker"""

    def __init__(
        self, max_age: int = None, min_hits: int = None, iou_threshold: float = None
    ):
        """
        Initialize tracker

        Args:
            max_age: Maximum frames to keep alive a track without detections
            min_hits: Minimum hits before track is confirmed
            iou_threshold: IOU threshold for matching
        """
        self.max_age = max_age or settings.TRACK_MAX_AGE
        self.min_hits = min_hits or settings.TRACK_MIN_HITS
        self.iou_threshold = iou_threshold or settings.TRACK_IOU_THRESHOLD

        self.trackers: List[KalmanBoxTracker] = []
        self.frame_count = 0

        print(
            f"[Tracker] Initialized (max_age={self.max_age}, min_hits={self.min_hits})"
        )

    def update(self, detections: List[Detection], frame_number: int) -> List[Track]:
        """
        Update tracker with new detections

        Args:
            detections: List of detections for current frame
            frame_number: Current frame number

        Returns:
            List of active tracks
        """
        self.frame_count = frame_number

        # Predict new locations
        for tracker in self.trackers:
            tracker.predict()

        # Match detections to trackers
        matched, unmatched_dets, unmatched_trks = self._match(detections)

        # Update matched trackers
        for det_idx, trk_idx in matched:
            det = detections[det_idx]
            self.trackers[trk_idx].update(det.bbox, det.class_id, det.class_name)

        # Create new trackers for unmatched detections
        for det_idx in unmatched_dets:
            det = detections[det_idx]
            tracker = KalmanBoxTracker(det.bbox, det.class_id, det.class_name)
            self.trackers.append(tracker)

        # Remove dead trackers
        self.trackers = [
            t for t in self.trackers if t.time_since_update <= self.max_age
        ]

        # Return confirmed tracks
        tracks = []
        for tracker in self.trackers:
            if tracker.hits >= self.min_hits or self.frame_count <= self.min_hits:
                bbox = tracker.get_state()
                track = Track(
                    track_id=tracker.id,
                    class_id=tracker.class_id,
                    class_name=tracker.class_name,
                    bbox=bbox,
                    confidence=0.9,  # High confidence for tracked objects
                    frame_number=frame_number,
                )
                tracks.append(track)

        return tracks

    def _match(self, detections: List[Detection]) -> Tuple[List, List, List]:
        """
        Match detections to trackers using Hungarian algorithm

        Returns:
            matched: List of (det_idx, trk_idx) pairs
            unmatched_detections: List of detection indices
            unmatched_trackers: List of tracker indices
        """
        if len(self.trackers) == 0:
            return [], list(range(len(detections))), []

        # Compute IOU cost matrix
        iou_matrix = np.zeros((len(detections), len(self.trackers)))
        for d, det in enumerate(detections):
            for t, trk in enumerate(self.trackers):
                iou_matrix[d, t] = self._iou(det.bbox, trk.get_state())

        # Hungarian algorithm (maximize IOU = minimize -IOU)
        if iou_matrix.size > 0:
            row_ind, col_ind = linear_sum_assignment(-iou_matrix)
            matched_indices = np.column_stack([row_ind, col_ind])
        else:
            matched_indices = np.empty((0, 2), dtype=int)

        # Filter out matches with low IOU
        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] >= self.iou_threshold:
                matches.append(m.tolist())

        matched = np.array(matches) if len(matches) > 0 else np.empty((0, 2), dtype=int)

        # Find unmatched detections and trackers
        unmatched_detections = [
            d for d in range(len(detections)) if d not in matched[:, 0]
        ]
        unmatched_trackers = [
            t for t in range(len(self.trackers)) if t not in matched[:, 1]
        ]

        return matched.tolist(), unmatched_detections, unmatched_trackers

    @staticmethod
    def _iou(bbox1: BoundingBox, bbox2: BoundingBox) -> float:
        """Calculate Intersection over Union"""
        # Intersection area
        x1 = max(bbox1.x1, bbox2.x1)
        y1 = max(bbox1.y1, bbox2.y1)
        x2 = min(bbox1.x2, bbox2.x2)
        y2 = min(bbox1.y2, bbox2.y2)

        intersection = max(0, x2 - x1) * max(0, y2 - y1)

        # Union area
        area1 = bbox1.width * bbox1.height
        area2 = bbox2.width * bbox2.height
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0

    def reset(self):
        """Reset tracker state"""
        self.trackers = []
        self.frame_count = 0
        KalmanBoxTracker.count = 0

    def visualize(self, frame: np.ndarray, tracks: List[Track]) -> np.ndarray:
        """Draw tracking results on frame"""
        frame_copy = frame.copy()

        for track in tracks:
            # Get coordinates
            x1, y1 = int(track.bbox.x1), int(track.bbox.y1)
            x2, y2 = int(track.bbox.x2), int(track.bbox.y2)

            # Draw box with unique color per track
            color = self._get_color(track.track_id)
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)

            # Draw track ID and class
            label = f"ID:{track.track_id} {track.class_name}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame_copy, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
            cv2.putText(
                frame_copy,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

        return frame_copy

    @staticmethod
    def _get_color(track_id: int) -> Tuple[int, int, int]:
        """Get consistent color for track ID"""
        np.random.seed(track_id)
        return tuple(np.random.randint(0, 255, 3).tolist())
