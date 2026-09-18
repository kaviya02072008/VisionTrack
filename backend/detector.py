import cv2

class PersonDetector:
    def __init__(self, model_path="yolov8n.pt", conf_threshold=0.45):
        """
        Initialize the YOLOv8 model for person detection and tracking.
        Uses YOLO's native ByteTrack implementation.
        """
        self.conf_threshold = conf_threshold
        self.person_class_id = 0  # Class 0 in COCO is 'person'
        self.model = None
        
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            print(f"Successfully loaded YOLOv8 model from {model_path}")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")

    def detect(self, frame):
        """
        Detect and track people in a video frame.
        Returns a list of structured dictionaries containing detection and tracking data.
        """
        detections = []
        
        if self.model is None or frame is None:
            return detections
            
        try:
            # Run tracking targeting only the person class (0) using ByteTrack
            results = self.model.track(
                source=frame, 
                classes=[self.person_class_id], 
                conf=self.conf_threshold, 
                tracker="bytetrack.yaml",
                persist=True,
                verbose=False
            )
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())
                    
                    # Extract tracking ID if assigned
                    track_id = int(box.id[0].cpu().numpy()) if box.id is not None else None
                    
                    if cls_id == self.person_class_id:
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)
                        
                        detections.append({
                            "track_id": track_id,
                            "x1": int(x1),
                            "y1": int(y1),
                            "x2": int(x2),
                            "y2": int(y2),
                            "center_x": center_x,
                            "center_y": center_y,
                            "confidence": round(conf, 2),
                            "class_name": "person"
                        })
                        
        except Exception as e:
            print(f"Error during detection/tracking: {e}")
            
        return detections

    def draw_detections(self, frame, detections):
        """
        Basic bounding box drawing (if tracking is disabled).
        Usually superseded by tracker.draw_tracking().
        """
        for det in detections:
            x1, y1 = det["x1"], det["y1"]
            x2, y2 = det["x2"], det["y2"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return frame
