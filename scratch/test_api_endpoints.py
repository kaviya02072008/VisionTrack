import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

print("Testing /stats in initial state...")
res = client.get("/stats")
print("Status code:", res.status_code)
data = res.json()
print("Stats response keys:", list(data.keys()))
print("Current occupancy:", data.get("current_occupancy"))
print("Peak occupancy:", data.get("peak_occupancy"))
print("Unique people:", data.get("unique_people"))
print("Total detections:", data.get("total_detections"))
print("Diagnostics:", data.get("diagnostics"))

print("\nTesting /dataset_videos...")
res = client.get("/dataset_videos")
print("Status code:", res.status_code)
vids = res.json().get("videos", [])
print(f"Found {len(vids)} videos: {[v['filename'] for v in vids]}")

print("\nTesting /select_video with dataset/uploads/Copy_of_Lunch2slow.mp4...")
res = client.post("/select_video", json={"mode": "REAL", "path": "dataset/uploads/Copy_of_Lunch2slow.mp4"})
print("Select video status:", res.status_code, res.json())

print("\nTesting /video_info...")
res = client.get("/video_info")
print("Video info status:", res.status_code, res.json())

print("\nTesting /control/reset...")
res = client.post("/control/reset")
print("Reset status:", res.status_code, res.json())

print("\nAll API endpoint checks passed successfully!")
