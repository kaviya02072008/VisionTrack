# VisionTrack — Intelligent Occupancy Pattern Analysis

VisionTrack is a real-time computer vision system that tracks people, analyzes indoor space utilization, builds dynamic spatial heatmaps, measures zone dwell times, and visualizes movement patterns across indoor environments.

---

## Problem Statement

Indoor space managers, retail operators, and venue planners often lack real-time visibility into how visitors navigate and utilize physical spaces. Traditional manual counting or basic sensors fail to capture continuous movement trails, dwell duration per area, high-traffic bottlenecks, and spatial transition patterns needed for data-driven decisions.

---

## Features

- **YOLOv8 Person Detection**: High-accuracy detection of individuals in surveillance feeds.
- **ByteTrack Person Tracking**: Multi-object tracking maintaining persistent track IDs and movement trails.
- **Occupancy Monitoring**: Real-time tracking of current, peak, and average occupancy.
- **Occupancy Heatmap**: Dynamic spatial heat accumulation representing spatial utilization over time.
- **Dwell Time Analysis**: Zone-level dwell tracking (Entrance, Main Floor, Checkout, Lounge) calculating minimum, maximum, and average duration.
- **Traffic Analysis**: Classification of zones into High, Medium, or Low traffic areas based on footfall and dwell profiles.
- **Movement Pattern Analysis**: Extraction of common paths and zone-to-zone transition matrices.
- **Dual Mode Operation**: 
  - **REAL VIDEO ANALYSIS**: Live processing of uploaded videos or dataset samples.
  - **DEMO MODE — SIMULATED DATA**: Synthetic deterministic fallback for instant demo readiness without video dependencies.
- **CSV Report Export**: Downloadable reports for tracking logs, occupancy timelines, zone statistics, and transition matrices.

---

## Technology Stack

- **Python 3.9+**
- **FastAPI & Uvicorn** (Backend REST API & MJPEG streaming)
- **OpenCV & NumPy** (Image processing & spatial heatmap accumulation)
- **YOLOv8 (Ultralytics)** (Person detection)
- **ByteTrack / Tracker** (Multi-object tracking)
- **Pandas** (CSV data aggregation & export)
- **Chart.js** (Dashboard visualization)
- **HTML5, CSS (TailwindCSS CDN), JavaScript** (Single-page web interface)

---

## Installation & Setup

1. **Navigate to Project Directory**:
   ```bash
   cd VisionTrack
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Run Command

Start the application with a single command:

```bash
py -3 backend/main.py
```

---

## Browser URL

Open your browser to access the dashboard:

```
http://127.0.0.1:8000
```

---

## Recommended Hackathon Demo Scenario

Follow this step-by-step sequence for a seamless live demonstration:

1. **Step 1 — Open VisionTrack**: Launch `http://127.0.0.1:8000` in Google Chrome or Microsoft Edge.
2. **Step 2 — Verify DEMO MODE**: VisionTrack starts immediately in `DEMO MODE — SIMULATED DATA`. Point out the header badge.
3. **Step 3 — Observe Live KPIs & Chart**: Show **Current Occupancy**, **Peak**, **Average**, and **Occupancy Over Time** updating live.
4. **Step 4 — Show Heatmap & Controls**: Demonstrate spatial heat accumulation, toggle **Hide Heatmap**, and test **Reset Heatmap**.
5. **Step 5 — Review Dwell Time & Traffic**: Highlight zone dwell stats (Entrance, Main Floor, Checkout, Lounge) and traffic intensity badges.
6. **Step 6 — Inspect Movement Patterns**: Point out the **Common Path** and the **Zone Transition Matrix**.
7. **Step 7 — Switch to REAL Video Mode**: Select a video from the **Select Dataset Video** dropdown (e.g., `dataset/sample.mp4`) and click **Load**.
8. **Step 8 — Start REAL Video Analysis**: Click **▶ Start Analysis**. Point out the updated mode badge (`REAL VIDEO ANALYSIS`), YOLO bounding boxes, ByteTrack IDs, and movement trails on the feed.
9. **Step 9 — Export Reports**: Click **📋 Tracking CSV** or **📈 Occupancy CSV** under Report Export to download timestamped CSV reports.
