import os
import time
import cv2
import numpy as np

class PersonDetector:
    def __init__(self, model_path="yolov8s.pt", conf_threshold=0.25, imgsz=1280):
        """
        Detect people using YOLOv8 with ByteTrack tracking.
        Optimized for surveillance, overhead angles, crowded scenes, and fisheye distortion.
        """
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        self.person_class_id = 0  # COCO class 0: person
        self.model = None
        self.model_name = "yolov8s.pt"
        self.frame_count = 0

        # Diagnostics storage
        self.last_diagnostics = {
            "detected_count": 0,
            "tracked_count": 0,
            "min_confidence": 0.0,
            "avg_confidence": 0.0,
            "max_confidence": 0.0,
            "inference_ms": 0.0,
            "inference_fps": 0.0,
            "model_name": self.model_name,
            "inference_imgsz": self.imgsz,
        }

        # Resolve model path: prefer yolov8s.pt, fallback to yolov8n.pt if missing
        resolved_path = self._resolve_model_path(model_path)
        if resolved_path is None and "yolov8s" in model_path:
            # Fallback to yolov8n.pt if yolov8s.pt not found
            resolved_path = self._resolve_model_path("yolov8n.pt")

        if resolved_path:
            self.model_name = os.path.basename(resolved_path)
            self.last_diagnostics["model_name"] = self.model_name

        try:
            from ultralytics import YOLO

            target = resolved_path or model_path
            self.model = YOLO(target)
            print(f"YOLO model loaded successfully: {target} (conf={self.conf_threshold}, imgsz={self.imgsz})")

            # Warm-up inference so first video frame runs at full speed
            self._warm_up()

        except Exception as e:
            print(f"Error loading YOLO model: {e}")

    def _resolve_model_path(self, path):
        """Find local model file across project directories."""
        if os.path.isabs(path) and os.path.exists(path):
            return path
        candidates = [
            os.path.abspath(path),
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), path),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), path),
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        return None

    def _warm_up(self):
        """Run a warm-up forward pass to initialize PyTorch CUDA/CPU kernels and buffers."""
        try:
            dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
            t0 = time.perf_counter()
            self.model.predict(source=dummy, classes=[self.person_class_id], conf=self.conf_threshold, imgsz=self.imgsz, verbose=False)
            dt = (time.perf_counter() - t0) * 1000
            print(f"[Detector] Warm-up complete in {dt:.1f}ms")
        except Exception as e:
            print(f"[Detector] Warm-up note: {e}")

    def reset_tracker(self):
        """Reset internal ByteTrack tracker states in YOLO predictor."""
        try:
            if self.model and hasattr(self.model, "predictor") and self.model.predictor:
                if hasattr(self.model.predictor, "trackers"):
                    for tr in self.model.predictor.trackers:
                        if hasattr(tr, "reset"):
                            tr.reset()
            self.frame_count = 0
            print("[Detector] Tracker state reset.")
        except Exception as e:
            print(f"[Detector] Note on tracker reset: {e}")

    def _suppress_duplicate_boxes(self, detections, iou_threshold=0.65):
        """
        Suppress duplicate overlapping detections on the same individual.
        Keeps the detection with higher confidence if IoU exceeds threshold.
        """
        if len(detections) <= 1:
            return detections

        # Sort descending by confidence
        sorted_dets = sorted(detections, key=lambda d: d["confidence"], reverse=True)
        keep = []

        for det in sorted_dets:
            duplicate = False
            for kept in keep:
                # Compute IoU
                x1 = max(det["x1"], kept["x1"])
                y1 = max(det["y1"], kept["y1"])
                x2 = min(det["x2"], kept["x2"])
                y2 = min(det["y2"], kept["y2"])
                inter = max(0, x2 - x1) * max(0, y2 - y1)
                a1 = (det["x2"] - det["x1"]) * (det["y2"] - det["y1"])
                a2 = (kept["x2"] - kept["x1"]) * (kept["y2"] - kept["y1"])
                iou = inter / float(a1 + a2 - inter + 1e-6)

                if iou > iou_threshold:
                    duplicate = True
                    break
            if not duplicate:
                keep.append(det)

        return keep

    def detect(self, frame):
        """
        Detect and track people in a video frame.
        Returns a list of dictionaries for PersonTracker.
        """
        self.frame_count += 1
        raw_detections = []

        if self.model is None or frame is None:
            return raw_detections

        t_start = time.perf_counter()

        try:
            results = self.model.track(
                source=frame,
                classes=[self.person_class_id],
                conf=self.conf_threshold,
                imgsz=self.imgsz,
                tracker="bytetrack.yaml",
                persist=True,
                verbose=False
            )

            for result in results:
                if result.boxes is None:
                    continue

                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())

                    if class_id != self.person_class_id:
                        continue

                    # Extract ByteTrack tracking ID if assigned
                    track_id = None
                    if box.id is not None:
                        track_id = int(box.id[0].cpu().numpy())

                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)

                    raw_detections.append({
                        "track_id": track_id,
                        "x1": int(x1),
                        "y1": int(y1),
                        "x2": int(x2),
                        "y2": int(y2),
                        "center_x": center_x,
                        "center_y": center_y,
                        "confidence": round(confidence, 3),
                        "class_name": "person"
                    })

            # Eliminate any duplicate overlapping detections
            detections = self._suppress_duplicate_boxes(raw_detections, iou_threshold=0.65)

            elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            inf_fps = round(1000.0 / max(1.0, elapsed_ms), 1)

            # Calculate and save diagnostic metrics
            confs = [d["confidence"] for d in detections]
            tracked_count = sum(d["track_id"] is not None for d in detections)
            min_c = min(confs) if confs else 0.0
            avg_c = round(sum(confs) / len(confs), 3) if confs else 0.0
            max_c = max(confs) if confs else 0.0

            self.last_diagnostics = {
                "detected_count": len(detections),
                "tracked_count": tracked_count,
                "min_confidence": min_c,
                "avg_confidence": avg_c,
                "max_confidence": max_c,
                "inference_ms": round(elapsed_ms, 1),
                "inference_fps": inf_fps,
                "model_name": self.model_name,
                "inference_imgsz": self.imgsz,
            }

            if detections:
                print(
                    f"[Detector] Frame {self.frame_count:4d}: {len(detections)} people detected "
                    f"(conf min={min_c:.3f}, avg={avg_c:.3f}, max={max_c:.3f}) | "
                    f"ByteTrack: {tracked_count}/{len(detections)} | {elapsed_ms:.1f}ms ({inf_fps} FPS)"
                )
            else:
                print(f"[Detector] Frame {self.frame_count:4d}: 0 people detected | {elapsed_ms:.1f}ms")

            return detections

        except Exception as e:
            print(f"[Detector] Detection/tracking error: {e}")
            return raw_detections

    def draw_detections(self, frame, detections):
        """Draw bounding boxes for detections. Scales line thickness with frame dimensions."""
        if frame is None or not detections:
            return frame

        scale = max(1.0, min(frame.shape[0], frame.shape[1]) / 640.0)
        thickness = max(2, int(round(scale * 1.5)))

        for det in detections:
            x1, y1 = det["x1"], det["y1"]
            x2, y2 = det["x2"], det["y2"]

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                thickness
            )

        return frame
