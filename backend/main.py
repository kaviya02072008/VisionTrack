import cv2
import threading
import time
import io
import os
import math
import datetime
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
from pydantic import BaseModel

from detector import PersonDetector
from tracker import PersonTracker
from analyzer import OccupancyAnalyzer
from heatmap import HeatmapGenerator

app = FastAPI(title="VisionTrack API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Global State ───────────────────────────────────────────────────────────
active_video_path = None
current_mode = "DEMO"
latest_frame = None
latest_heatmap = None
latest_stats = {}
is_running = True      # analysis loop is always running (DEMO by default)
tracking_log = []      # accumulated per-object rows for CSV export
state_lock = threading.Lock()

# Shared references so control endpoints can call reset on live objects
_analyzer_ref = None
_tracker_ref = None
_heatmap_ref = None

MAX_TRACKING_LOG = 50000  # Cap tracking log to prevent OOM

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "..", "dataset")
UPLOADS_DIR = os.path.join(DATASET_DIR, "uploads")
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")

os.makedirs(UPLOADS_DIR, exist_ok=True)
ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov"}


# ─── Demo frame generator ───────────────────────────────────────────────────
def get_demo_frame(frame_count):
    demo_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    demo_frame[:] = (50, 50, 100)
    cv2.putText(
        demo_frame,
        "DEMO MODE - SIMULATED DATA",
        (50, 240),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2,
    )

    # Two deterministic sine-wave trajectories
    cx1 = int(320 + math.sin(frame_count * 0.05) * 150)
    cy1 = int(240 + math.cos(frame_count * 0.05) * 100)
    cx2 = int(320 + math.sin(frame_count * 0.03 + 2) * 200)
    cy2 = int(240 + math.cos(frame_count * 0.04 + 1) * 80)

    tracked_objects = [
        {
            "track_id": "demo_1",
            "center_x": cx1,
            "center_y": cy1,
            "x1": cx1 - 20,
            "y1": cy1 - 40,
            "x2": cx1 + 20,
            "y2": cy1 + 40,
            "confidence": 1.0,
            "class_name": "person",
        },
        {
            "track_id": "demo_2",
            "center_x": cx2,
            "center_y": cy2,
            "x1": cx2 - 20,
            "y1": cy2 - 40,
            "x2": cx2 + 20,
            "y2": cy2 + 40,
            "confidence": 1.0,
            "class_name": "person",
        },
    ]
    return demo_frame, tracked_objects


# ─── Processing thread ───────────────────────────────────────────────────────
def video_processing_thread():
    global latest_frame, latest_heatmap, latest_stats
    global current_mode, active_video_path
    global _analyzer_ref, _tracker_ref, _heatmap_ref, tracking_log

    detector = PersonDetector()
    tracker = PersonTracker()
    analyzer = OccupancyAnalyzer()
    heatmap_gen = HeatmapGenerator(640, 480)

    with state_lock:
        _analyzer_ref = analyzer
        _tracker_ref = tracker
        _heatmap_ref = heatmap_gen

    cap = None
    last_video_path = None
    demo_frame_count = 0

    while True:
        with state_lock:
            mode = current_mode
            video_path = active_video_path
            running = is_running

        if not running:
            time.sleep(0.2)
            continue

        # ── Video changed → reset analytics ─────────────────────────────────
        if video_path != last_video_path:
            analyzer.reset()
            tracker.reset()
            with state_lock:
                tracking_log.clear()
            if cap:
                cap.release()
                cap = None
            last_video_path = video_path

        # ── DEMO MODE ────────────────────────────────────────────────────────
        if mode == "DEMO" or not video_path:
            demo_frame_count += 1
            frame, tracked_objects = get_demo_frame(demo_frame_count)

            if analyzer.frame_width is None:
                analyzer.update_frame_info(640, 480, 30)
            if heatmap_gen.width != 640 or heatmap_gen.height != 480:
                heatmap_gen = HeatmapGenerator(640, 480)
                with state_lock:
                    _heatmap_ref = heatmap_gen

            stats = analyzer.update(tracked_objects)
            stats["is_demo"] = True

            heatmap_gen.update(tracked_objects)
            hm_frame = heatmap_gen.get_heatmap_image(frame)

            # Log for CSV
            ts = round(demo_frame_count / 30.0, 3)
            with state_lock:
                for obj in tracked_objects:
                    zone = analyzer._detect_zone(obj["center_x"], obj["center_y"])
                    tracking_log.append({
                        "frame": analyzer.frame_count,
                        "timestamp": ts,
                        "track_id": obj["track_id"],
                        "center_x": obj["center_x"],
                        "center_y": obj["center_y"],
                        "confidence": obj["confidence"],
                        "zone": zone or "",
                        "source": "demo",
                    })
                if len(tracking_log) > MAX_TRACKING_LOG:
                    tracking_log[:] = tracking_log[-MAX_TRACKING_LOG:]

            with state_lock:
                latest_frame = frame
                latest_heatmap = hm_frame
                latest_stats = stats

            time.sleep(0.1)
            continue

        # ── REAL MODE ────────────────────────────────────────────────────────
        if cap is None:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30.0
            if not cap.isOpened():
                print(f"Error opening {video_path}. Switching to DEMO.")
                with state_lock:
                    current_mode = "DEMO"
                    active_video_path = None
                cap = None
                continue

        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            time.sleep(0.03)
            continue

        h, w = frame.shape[:2]
        analyzer.update_frame_info(w, h, fps)
        if heatmap_gen.width != w or heatmap_gen.height != h:
            heatmap_gen = HeatmapGenerator(w, h)
            with state_lock:
                _heatmap_ref = heatmap_gen

        # Pipeline: detect → track → analyze → heatmap → draw
        detections = detector.detect(frame)
        tracked_objects = tracker.update(detections, frame)
        stats = analyzer.update(tracked_objects)
        stats["is_demo"] = False

        heatmap_gen.update(tracked_objects)
        hm_frame = heatmap_gen.get_heatmap_image(frame)
        frame = tracker.draw_tracking(frame, tracked_objects)

        # Log for CSV
        ts = round(analyzer.frame_count / fps, 3)
        with state_lock:
            for obj in tracked_objects:
                zone = analyzer._detect_zone(obj["center_x"], obj["center_y"])
                tracking_log.append({
                    "frame": analyzer.frame_count,
                    "timestamp": ts,
                    "track_id": obj["track_id"],
                    "center_x": obj["center_x"],
                    "center_y": obj["center_y"],
                    "confidence": round(obj.get("confidence", 0.0), 3),
                    "zone": zone or "",
                    "source": "real",
                })
            if len(tracking_log) > MAX_TRACKING_LOG:
                tracking_log[:] = tracking_log[-MAX_TRACKING_LOG:]

        with state_lock:
            latest_frame = frame
            latest_heatmap = hm_frame
            latest_stats = stats

        time.sleep(1 / 30.0)


# ─── MJPEG stream generator ──────────────────────────────────────────────────
def frame_generator(feed_type="standard"):
    while True:
        with state_lock:
            frame = latest_heatmap if feed_type == "heatmap" else latest_frame

        if frame is not None:
            ret, buffer = cv2.imencode(".jpg", frame)
            if ret:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + buffer.tobytes()
                    + b"\r\n"
                )
        time.sleep(0.03)


# ─── Startup ─────────────────────────────────────────────────────────────────
@app.on_event("startup")
def startup_event():
    thread = threading.Thread(target=video_processing_thread, daemon=True)
    thread.start()


# ─── Existing endpoints (unchanged) ──────────────────────────────────────────
@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        frame_generator("standard"), media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/heatmap_feed")
def heatmap_feed():
    return StreamingResponse(
        frame_generator("heatmap"), media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/stats")
def get_stats():
    with state_lock:
        stats = dict(latest_stats)
        stats["is_running"] = is_running
    return stats


@app.post("/upload_video")
async def upload_video(file: UploadFile = File(...)):
    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()

    if ext not in ALLOWED_EXTENSIONS:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": f"Unsupported format '{ext}'. Allowed: .mp4, .avi, .mov"},
        )

    safe_filename = os.path.basename(filename).replace(" ", "_")
    filepath = os.path.join(UPLOADS_DIR, safe_filename)

    with open(filepath, "wb") as f:
        f.write(await file.read())

    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        os.remove(filepath)
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": "File appears corrupt or is not a valid video."},
        )
    cap.release()

    return {"success": True, "filename": safe_filename, "path": f"dataset/uploads/{safe_filename}"}


@app.get("/dataset_videos")
def get_dataset_videos():
    videos = []
    if os.path.exists(DATASET_DIR):
        for f in os.listdir(DATASET_DIR):
            ext = os.path.splitext(f)[1].lower()
            if ext in ALLOWED_EXTENSIONS and os.path.isfile(os.path.join(DATASET_DIR, f)):
                videos.append({"filename": f, "path": f"dataset/{f}"})

    if os.path.exists(UPLOADS_DIR):
        for f in os.listdir(UPLOADS_DIR):
            ext = os.path.splitext(f)[1].lower()
            if ext in ALLOWED_EXTENSIONS and os.path.isfile(os.path.join(UPLOADS_DIR, f)):
                videos.append({"filename": f, "path": f"dataset/uploads/{f}"})

    return {"videos": videos}


class SelectVideoReq(BaseModel):
    mode: str = "REAL"
    path: str = ""


@app.post("/select_video")
def select_video(req: SelectVideoReq):
    global current_mode, active_video_path

    if req.mode == "DEMO":
        with state_lock:
            current_mode = "DEMO"
            active_video_path = None
        return {"success": True, "mode": "DEMO"}

    if not req.path.startswith("dataset/") and not req.path.startswith("dataset\\"):
        return JSONResponse(status_code=400, content={"success": False, "error": "Invalid path."})

    norm_path = req.path.replace("\\", "/")
    full_path = os.path.normpath(os.path.join(BASE_DIR, "..", norm_path))

    if not os.path.exists(full_path):
        return JSONResponse(status_code=404, content={"success": False, "error": "File not found."})

    cap = cv2.VideoCapture(full_path)
    if not cap.isOpened():
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": "OpenCV cannot open this video file."},
        )
    cap.release()

    with state_lock:
        current_mode = "REAL"
        active_video_path = full_path

    return {"success": True, "mode": "REAL", "path": req.path}


@app.get("/video_info")
def get_video_info():
    with state_lock:
        path = active_video_path
        mode = current_mode

    if mode == "DEMO" or not path:
        return JSONResponse(status_code=400, content={"error": "No video selected or in DEMO mode."})

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return JSONResponse(status_code=400, content={"error": "Invalid video."})

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    duration = 0.0 if fps <= 0 else frame_count / fps

    return {
        "filename": os.path.basename(path),
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration_seconds": duration,
    }


# ─── Control endpoints ────────────────────────────────────────────────────────
@app.post("/control/start")
def control_start():
    """Start analysis. If a real video is selected, switches to REAL mode."""
    global is_running, current_mode
    with state_lock:
        is_running = True
        # If a video path is available, force REAL mode
        if active_video_path:
            current_mode = "REAL"
        mode = current_mode
    return {"success": True, "is_running": True, "mode": mode}


@app.post("/control/stop")
def control_stop():
    """Pause the analysis loop. State and logs are preserved."""
    global is_running
    with state_lock:
        is_running = False
    return {"success": True, "is_running": False}


@app.post("/control/reset")
def control_reset():
    """Reset analytics only. Keeps selected video and current mode."""
    global tracking_log, latest_stats
    if _analyzer_ref is not None:
        _analyzer_ref.reset()
    if _tracker_ref is not None:
        _tracker_ref.reset()
    with state_lock:
        tracking_log.clear()
        latest_stats = {}
        is_running_now = is_running
        mode = current_mode
    return {"success": True, "mode": mode, "is_running": is_running_now}


@app.post("/control/reset_heatmap")
def control_reset_heatmap():
    """Clear heatmap accumulation only. Does not affect tracking, occupancy, or dwell data."""
    if _heatmap_ref is not None:
        _heatmap_ref.reset()
    return {"success": True, "message": "Heatmap accumulation cleared."}


# ─── CSV export helpers ───────────────────────────────────────────────────────
def _ts_suffix():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def _csv_response(df: pd.DataFrame, filename: str):
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/export/tracking_csv")
def export_tracking_csv():
    with state_lock:
        log_copy = list(tracking_log)
        mode = current_mode

    prefix = "demo_" if mode == "DEMO" else ""
    filename = f"{prefix}tracking_{_ts_suffix()}.csv"

    if not log_copy:
        df = pd.DataFrame(columns=["frame", "timestamp", "track_id", "center_x", "center_y", "confidence", "zone", "source"])
    else:
        df = pd.DataFrame(log_copy)

    return _csv_response(df, filename)


@app.get("/export/occupancy_csv")
def export_occupancy_csv():
    with state_lock:
        history = list(latest_stats.get("history", []))
        mode = current_mode

    prefix = "demo_" if mode == "DEMO" else ""
    filename = f"{prefix}occupancy_{_ts_suffix()}.csv"

    source = "demo" if mode == "DEMO" else "real"
    fps = 30.0
    if _analyzer_ref and _analyzer_ref.fps:
        fps = _analyzer_ref.fps

    rows = []
    for h in history:
        rows.append({
            "frame": h["frame"],
            "timestamp": round(h["frame"] / fps, 3),
            "occupancy": h["occupancy"],
            "source": source,
        })

    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["frame", "timestamp", "occupancy", "source"])
    return _csv_response(df, filename)


@app.get("/export/zone_analysis_csv")
def export_zone_analysis_csv():
    with state_lock:
        mode = current_mode

    prefix = "demo_" if mode == "DEMO" else ""
    filename = f"{prefix}zone_analysis_{_ts_suffix()}.csv"
    source = "demo" if mode == "DEMO" else "real"

    rows = []
    if _analyzer_ref:
        fps = _analyzer_ref.fps or 30.0
        for zone, times in _analyzer_ref.zone_visits.items():
            visits = len(times)
            times_sec = [round(t / fps, 3) for t in times]
            total = round(sum(times_sec), 3)
            avg = round(total / visits, 3) if visits else 0.0
            traffic = _analyzer_ref.classify_traffic(visits)
            rows.append({
                "zone": zone,
                "visits": visits,
                "total_dwell_seconds": total,
                "average_dwell_seconds": avg,
                "min_dwell_seconds": round(min(times_sec), 3) if times_sec else 0,
                "max_dwell_seconds": round(max(times_sec), 3) if times_sec else 0,
                "current_occupancy": _analyzer_ref.current_occupancy,
                "traffic_level": traffic,
                "source": source,
            })

    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["zone", "visits", "total_dwell_seconds", "average_dwell_seconds",
                 "min_dwell_seconds", "max_dwell_seconds", "current_occupancy", "traffic_level", "source"]
    )
    return _csv_response(df, filename)


@app.get("/export/movement_matrix_csv")
def export_movement_matrix_csv():
    with state_lock:
        mode = current_mode

    prefix = "demo_" if mode == "DEMO" else ""
    filename = f"{prefix}movement_matrix_{_ts_suffix()}.csv"
    source = "demo" if mode == "DEMO" else "real"

    rows = []
    if _analyzer_ref:
        for tid, zones in _analyzer_ref.track_transitions.items():
            for i in range(len(zones) - 1):
                from_z = zones[i]
                to_z = zones[i + 1]
                # Find or create row
                found = False
                for r in rows:
                    if r["from_zone"] == from_z and r["to_zone"] == to_z:
                        r["movement_count"] += 1
                        found = True
                        break
                if not found:
                    rows.append({"from_zone": from_z, "to_zone": to_z, "movement_count": 1, "source": source})

    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["from_zone", "to_zone", "movement_count", "source"]
    )
    return _csv_response(df, filename)


# ─── Serve frontend ───────────────────────────────────────────────────────────
if os.path.exists(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
else:
    print(f"Warning: frontend not found at {FRONTEND_DIR}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
