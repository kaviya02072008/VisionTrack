import os
import sys
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.detector import PersonDetector
from backend.tracker import PersonTracker
from backend.analyzer import OccupancyAnalyzer

video_path = os.path.join(BASE_DIR, "dataset", "uploads", "Copy_of_Lunch2slow.mp4")
model_path = os.path.join(BASE_DIR, "yolov8s.pt")
if not os.path.exists(model_path):
    model_path = os.path.join(BASE_DIR, "yolov8n.pt")

print(f"Using model: {model_path}")
print(f"Using video: {video_path}")

cap = cv2.VideoCapture(video_path)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 9.0

detector = PersonDetector(model_path=model_path, conf_threshold=0.25, imgsz=960)
tracker = PersonTracker()
analyzer = OccupancyAnalyzer()
analyzer.update_frame_info(w, h, fps)

unique_bytetrack_ids = set()
frames_to_test = 60

for f in range(frames_to_test):
    ret, frame = cap.read()
    if not ret:
        break
    dets = detector.detect(frame)
    tracked = tracker.update(dets, frame)
    stats = analyzer.update(tracked)
    for obj in tracked:
        unique_bytetrack_ids.add(obj["track_id"])
    
    if (f + 1) % 5 == 0:
        print(f"Frame {f+1:2d} | Visible in frame: {len(tracked):2d} | Current Occ: {stats['current_occupancy']:2d} | Unique in Analyzer: {stats['unique_people']:2d} | Raw ByteTrack IDs so far: {len(unique_bytetrack_ids):2d}")

cap.release()
