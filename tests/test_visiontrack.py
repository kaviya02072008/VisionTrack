import os
import sys
import unittest
import numpy as np
import cv2

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.detector import PersonDetector
from backend.tracker import PersonTracker
from backend.analyzer import OccupancyAnalyzer
from backend.heatmap import HeatmapGenerator


class TestVisionTrackComponents(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_path = os.path.join(PROJECT_ROOT, "yolov8n.pt")
        cls.video_path = os.path.join(PROJECT_ROOT, "dataset", "uploads", "Copy_of_Lunch2slow.mp4")

    def test_01_detector_initialization(self):
        detector = PersonDetector(model_path=self.model_path, conf_threshold=0.20, imgsz=960)
        self.assertIsNotNone(detector.model, "YOLOv8 model failed to load")
        self.assertEqual(detector.person_class_id, 0, "COCO class ID must be 0 for person")
        self.assertEqual(detector.conf_threshold, 0.20)
        self.assertEqual(detector.imgsz, 960)

    def test_02_detector_reset_tracker(self):
        detector = PersonDetector(model_path=self.model_path)
        # Should not raise exception
        detector.reset_tracker()
        self.assertEqual(detector.frame_count, 0)

    def test_03_tracker_missing_ids_not_discarded(self):
        """Verify that detections with missing tracking IDs (None) are NOT discarded."""
        tracker = PersonTracker()
        dummy_detections = [
            {"track_id": 1, "x1": 100, "y1": 100, "x2": 150, "y2": 200, "center_x": 125, "center_y": 150, "confidence": 0.8},
            {"track_id": None, "x1": 300, "y1": 300, "x2": 350, "y2": 400, "center_x": 325, "center_y": 350, "confidence": 0.45},
            {"track_id": None, "x1": 500, "y1": 500, "x2": 550, "y2": 600, "center_x": 525, "center_y": 550, "confidence": 0.35},
        ]
        dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        tracked = tracker.update(dummy_detections, dummy_frame)
        self.assertEqual(len(tracked), 3, f"Expected 3 tracked objects, got {len(tracked)}. Detections were incorrectly discarded!")
        # Check that fallback IDs are assigned
        ids = [obj["track_id"] for obj in tracked]
        self.assertIn("1", ids)
        self.assertTrue(any(i.startswith("u") for i in ids), "Fallback IDs should be assigned for None track_ids")

    def test_04_tracker_spatial_continuity(self):
        """Verify that a track whose ByteTrack ID is lost on the next frame is associated spatially."""
        tracker = PersonTracker()
        dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        
        # Frame 1: Detection with ByteTrack ID 5
        frame1_dets = [
            {"track_id": 5, "x1": 200, "y1": 200, "x2": 250, "y2": 300, "center_x": 225, "center_y": 250, "confidence": 0.75}
        ]
        tracked1 = tracker.update(frame1_dets, dummy_frame)
        self.assertEqual(tracked1[0]["track_id"], "5")

        # Frame 2: Same person moved slightly (228, 252), but ByteTrack ID is None
        frame2_dets = [
            {"track_id": None, "x1": 203, "y1": 202, "x2": 253, "y2": 302, "center_x": 228, "center_y": 252, "confidence": 0.60}
        ]
        tracked2 = tracker.update(frame2_dets, dummy_frame)
        self.assertEqual(tracked2[0]["track_id"], "5", "Spatial association should maintain ID 5 across momentary loss")

    def test_05_analyzer_metrics(self):
        analyzer = OccupancyAnalyzer()
        analyzer.update_frame_info(640, 480, fps=30.0)

        tracked_objects = [
            {"track_id": "1", "center_x": 100, "center_y": 200, "confidence": 0.8},
            {"track_id": "2", "center_x": 300, "center_y": 300, "confidence": 0.7},
        ]
        stats = analyzer.update(tracked_objects)
        self.assertEqual(stats["current_occupancy"], 2)
        self.assertEqual(stats["peak_occupancy"], 2)
        self.assertEqual(stats["unique_people"], 2)
        self.assertIn("dwell_stats", stats)
        self.assertIn("traffic_stats", stats)

    def test_06_heatmap_generation(self):
        heatmap = HeatmapGenerator(640, 480)
        tracked_objects = [
            {"center_x": 320, "center_y": 240},
            {"center_x": 100, "center_y": 100},
        ]
        heatmap.update(tracked_objects)
        self.assertTrue(np.max(heatmap.heatmap_layer) > 0, "Heatmap layer should accumulate heat")

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        overlay = heatmap.get_heatmap_image(frame)
        self.assertEqual(overlay.shape, (480, 640, 3))
        # Ensure reset works
        heatmap.reset()
        self.assertEqual(np.max(heatmap.heatmap_layer), 0.0)

    def test_07_real_video_detection_and_tracking(self):
        """Test on actual surveillance footage if available."""
        if not os.path.exists(self.video_path):
            self.skipTest(f"Video {self.video_path} not found")

        cap = cv2.VideoCapture(self.video_path)
        self.assertTrue(cap.isOpened(), "Could not open real surveillance video")

        detector = PersonDetector(model_path=self.model_path, conf_threshold=0.20, imgsz=960)
        tracker = PersonTracker()
        analyzer = OccupancyAnalyzer()
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        analyzer.update_frame_info(w, h, fps)
        heatmap = HeatmapGenerator(w, h)

        # Process 5 frames
        total_detected = 0
        total_tracked = 0
        for _ in range(5):
            ret, frame = cap.read()
            if not ret:
                break
            dets = detector.detect(frame)
            tracked = tracker.update(dets, frame)
            stats = analyzer.update(tracked)
            heatmap.update(tracked)
            total_detected += len(dets)
            total_tracked += len(tracked)

        cap.release()
        self.assertGreater(total_detected, 0, "Real video should detect people")
        self.assertEqual(total_detected, total_tracked, "No detections should be discarded by tracker")
        self.assertGreater(stats["current_occupancy"], 1, "Should detect multiple people in real surveillance video")

    def test_08_metric_separation_and_reid(self):
        """Verify metric separation: current_occupancy drops on exit, unique_people preserved, total_detections increments."""
        analyzer = OccupancyAnalyzer()
        analyzer.update_frame_info(1920, 1080, fps=30.0)

        # Frame 1: 2 people visible
        f1 = [
            {"track_id": "1", "center_x": 300, "center_y": 400, "confidence": 0.8},
            {"track_id": "2", "center_x": 800, "center_y": 500, "confidence": 0.8},
        ]
        s1 = analyzer.update(f1)
        self.assertEqual(s1["current_occupancy"], 2)
        self.assertEqual(s1["unique_people"], 2)
        self.assertEqual(s1["total_detections"], 2)

        # Frame 2: Person 2 leaves, only Person 1 remains
        f2 = [
            {"track_id": "1", "center_x": 305, "center_y": 402, "confidence": 0.8},
        ]
        s2 = analyzer.update(f2)
        self.assertEqual(s2["current_occupancy"], 1, "Current occupancy must drop when a person leaves")
        self.assertEqual(s2["unique_people"], 2, "Unique people must remain preserved across session")
        self.assertEqual(s2["total_detections"], 3, "Total detections debug count must increment")

        # Frame 3: ByteTrack drops Person 1's ID and re-assigns ID 99 at same spot
        f3 = [
            {"track_id": "99", "center_x": 308, "center_y": 405, "confidence": 0.75},
        ]
        s3 = analyzer.update(f3)
        self.assertEqual(s3["current_occupancy"], 1)
        self.assertEqual(s3["unique_people"], 2, "Spatial Re-ID should prevent duplicate unique identity on ID switch")
        self.assertEqual(s3["total_detections"], 4)

    def test_09_no_duplicate_counting_for_static_tracks(self):
        """Verify that unique people count does NOT increase when the same person remains visible for 50 frames."""
        analyzer = OccupancyAnalyzer()
        analyzer.update_frame_info(1920, 1080, fps=30.0)

        for i in range(50):
            objs = [
                {"track_id": "1", "center_x": 200, "center_y": 300, "confidence": 0.85},
                {"track_id": "2", "center_x": 600, "center_y": 400, "confidence": 0.80},
            ]
            stats = analyzer.update(objs)
            self.assertEqual(stats["current_occupancy"], 2)
            self.assertEqual(stats["unique_people"], 2, f"Unique people must stay at 2, got {stats['unique_people']} on frame {i+1}")
            self.assertEqual(stats["total_detections"], (i + 1) * 2)


if __name__ == "__main__":
    unittest.main()

