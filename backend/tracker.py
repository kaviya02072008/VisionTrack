import cv2
import time
import math
from collections import deque

class PersonTracker:
    def __init__(self, max_history=50):
        """
        Maintains tracking history, movement trails, and metadata.
        Uses tracking IDs assigned by YOLOv8 ByteTrack, with robust spatial fallback
        for unconfirmed or missing IDs so no genuine detections are discarded.
        max_history limits the trail length for drawing.
        """
        self.trajectories = {}  # track_id -> deque of (center_x, center_y)
        self.last_seen = {}     # track_id -> frame_count (for cleanup)
        self.velocities = {}    # track_id -> (vx, vy) per frame
        self.max_history = max_history
        self.frame_count = 0
        self._fallback_id_counter = 0

    def reset(self):
        """Clear all trajectory history, counters, and fallback IDs."""
        self.trajectories.clear()
        self.last_seen.clear()
        self.velocities.clear()
        self.frame_count = 0
        self._fallback_id_counter = 0

    def update(self, detections, frame=None):
        """
        Update tracker history using detections from PersonDetector.
        detections: list of dicts from PersonDetector (may contain track_id)
        frame: optional frame for dimension-aware matching
        Returns:
            list of tracked objects with guaranteed persistent track_ids
        """
        self.frame_count += 1
        current_time = time.time()
        tracked_objects = []

        if not detections:
            self._cleanup_stale()
            return tracked_objects

        # Determine frame dimensions for distance scaling
        frame_h, frame_w = (1080, 1920)
        if frame is not None and hasattr(frame, "shape"):
            frame_h, frame_w = frame.shape[:2]
        max_match_dist = max(60, int(min(frame_w, frame_h) * 0.08))

        assigned_in_this_frame = set()
        unassigned_dets = []

        # Step 1: Process detections that already have a ByteTrack ID
        for det in detections:
            track_id = det.get("track_id")
            if track_id is not None:
                track_id_str = str(track_id)
                assigned_in_this_frame.add(track_id_str)

                cx = det["center_x"]
                cy = det["center_y"]

                if track_id_str not in self.trajectories:
                    self.trajectories[track_id_str] = deque(maxlen=self.max_history)

                # Compute velocity delta
                if self.trajectories[track_id_str]:
                    last_pt = self.trajectories[track_id_str][-1]
                    self.velocities[track_id_str] = (cx - last_pt[0], cy - last_pt[1])
                else:
                    self.velocities[track_id_str] = (0, 0)

                self.trajectories[track_id_str].append((cx, cy))
                self.last_seen[track_id_str] = self.frame_count

                obj = dict(det)
                obj["track_id"] = track_id_str
                obj["frame_number"] = self.frame_count
                obj["timestamp"] = current_time
                obj["vx"], obj["vy"] = self.velocities[track_id_str]
                tracked_objects.append(obj)
            else:
                unassigned_dets.append(det)

        # Step 2: For detections without a ByteTrack ID, associate spatially with recent active tracks
        # or assign a persistent fallback ID (DO NOT discard)
        for det in unassigned_dets:
            cx = det["center_x"]
            cy = det["center_y"]

            best_tid = None
            min_dist = float("inf")

            # Check distance to active tracks not yet updated in this frame
            for tid, pts in self.trajectories.items():
                if tid in assigned_in_this_frame:
                    continue
                # Only associate with tracks seen within last 20 frames
                if self.frame_count - self.last_seen.get(tid, 0) > 20:
                    continue
                last_pt = pts[-1]
                dist = math.hypot(cx - last_pt[0], cy - last_pt[1])
                if dist < max_match_dist and dist < min_dist:
                    min_dist = dist
                    best_tid = tid

            if best_tid is not None:
                track_id_str = best_tid
            else:
                self._fallback_id_counter += 1
                track_id_str = f"u{self._fallback_id_counter}"

            assigned_in_this_frame.add(track_id_str)

            if track_id_str not in self.trajectories:
                self.trajectories[track_id_str] = deque(maxlen=self.max_history)

            if self.trajectories[track_id_str]:
                last_pt = self.trajectories[track_id_str][-1]
                self.velocities[track_id_str] = (cx - last_pt[0], cy - last_pt[1])
            else:
                self.velocities[track_id_str] = (0, 0)

            self.trajectories[track_id_str].append((cx, cy))
            self.last_seen[track_id_str] = self.frame_count

            obj = dict(det)
            obj["track_id"] = track_id_str
            obj["frame_number"] = self.frame_count
            obj["timestamp"] = current_time
            obj["vx"], obj["vy"] = self.velocities[track_id_str]
            tracked_objects.append(obj)

        self._cleanup_stale()
        return tracked_objects

    def _cleanup_stale(self):
        """Clean up stale trajectories to prevent memory leaks (keep for ~100 frames)."""
        stale_ids = [
            tid for tid, last_frame in self.last_seen.items()
            if self.frame_count - last_frame > 100
        ]
        for tid in stale_ids:
            self.trajectories.pop(tid, None)
            self.last_seen.pop(tid, None)
            self.velocities.pop(tid, None)

    def draw_tracking(self, frame, tracked_objects, dt_frames=0):
        """
        Draw bounding boxes, tracking IDs, and movement trails on the frame.
        Adapts scale, stroke thickness, and font sizes to frame dimensions.
        dt_frames: optional frame offset to project positions during intermediate playback frames.
        """
        if frame is None or not tracked_objects:
            return frame

        # Adaptive scaling for high-resolution surveillance feeds (e.g., 2048x2048)
        scale = max(1.0, min(frame.shape[0], frame.shape[1]) / 640.0)
        thickness = max(2, int(round(scale * 1.5)))
        trail_thickness = max(2, int(round(scale * 1.2)))
        font_scale = max(0.45, 0.45 * scale)
        text_thickness = max(1, int(round(scale * 1.0)))
        label_padding = max(20, int(round(25 * scale)))

        frame_h, frame_w = frame.shape[:2]

        for obj in tracked_objects:
            track_id = obj["track_id"]

            # Optional velocity projection for intermediate sampled frames
            dx = int(obj.get("vx", 0) * dt_frames)
            dy = int(obj.get("vy", 0) * dt_frames)

            x1 = max(0, min(frame_w - 1, obj["x1"] + dx))
            y1 = max(0, min(frame_h - 1, obj["y1"] + dy))
            x2 = max(0, min(frame_w - 1, obj["x2"] + dx))
            y2 = max(0, min(frame_h - 1, obj["y2"] + dy))

            # Draw bounding box (Yellow)
            color = (0, 255, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            # Display ID badge with background
            conf_str = f" {obj['confidence']:.2f}" if "confidence" in obj else ""
            label = f"ID {track_id}{conf_str}"
            (w, h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)
            label_y = max(y1, label_padding)
            bg_top = max(0, label_y - h - int(5 * scale))
            cv2.rectangle(frame, (x1, bg_top), (x1 + w + int(6 * scale), label_y), color, -1)
            cv2.putText(
                frame,
                label,
                (x1 + int(3 * scale), label_y - int(3 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                text_thickness
            )

            # Draw movement trail (Magenta)
            if track_id in self.trajectories:
                trail = list(self.trajectories[track_id])
                for i in range(1, len(trail)):
                    pt1 = trail[i - 1]
                    pt2 = trail[i]
                    cv2.line(frame, pt1, pt2, (255, 0, 255), trail_thickness)

        return frame
