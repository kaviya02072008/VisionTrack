import numpy as np
import cv2

class HeatmapGenerator:
    def __init__(self, frame_width, frame_height):
        self.width = frame_width
        self.height = frame_height
        self.reset()
        
    def reset(self):
        # Accumulate indefinitely without decay to build a true occupancy heatmap
        self.heatmap_layer = np.zeros((self.height, self.width), dtype=np.float32)
        
    def update(self, tracked_objects):
        # Adapt hotspot radius proportionally to frame dimensions
        radius = max(25, int(min(self.width, self.height) * 0.025))

        for obj in tracked_objects:
            cx, cy = obj["center_x"], obj["center_y"]
            
            if 0 <= cx < self.width and 0 <= cy < self.height:
                # Bounding box for the circle
                min_y = max(0, cy - radius)
                max_y = min(self.height, cy + radius + 1)
                min_x = max(0, cx - radius)
                max_x = min(self.width, cx + radius + 1)
                
                # Create coordinate grids
                y, x = np.ogrid[min_y:max_y, min_x:max_x]
                
                # Circular mask
                mask = (x - cx)**2 + (y - cy)**2 <= radius**2
                
                # Add heat (strength 1)
                self.heatmap_layer[min_y:max_y, min_x:max_x][mask] += 1.0
                
    def get_heatmap_image(self, frame):
        if frame is None:
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # Resize frame if dimensions don't match heatmap
        h, w = frame.shape[:2]
        if h != self.height or w != self.width:
            frame = cv2.resize(frame, (self.width, self.height))
        max_val = np.max(self.heatmap_layer)
        if max_val == 0:
            return frame.copy() # No heat yet
            
        # Cap normalization at 80% of max, but ensure at least 10 hits for full red
        cap = max(10.0, max_val * 0.8)
        
        # Normalize to 0-1 range
        norm = np.clip(self.heatmap_layer / cap, 0, 1)
        
        # Convert to 8-bit
        norm_8u = (norm * 255).astype(np.uint8)
        
        # Apply JET colormap: 0=Blue, 255=Red
        color_heatmap = cv2.applyColorMap(norm_8u, cv2.COLORMAP_JET)
        
        # Create alpha blending mask
        # 0 heat -> 0 alpha (transparent)
        # Max heat -> 0.6 alpha (semi-transparent)
        alpha_mask = norm * 0.6
        alpha_mask_3d = np.repeat(alpha_mask[:, :, np.newaxis], 3, axis=2)
        
        # Ensure frame is uint8 float for blending, then convert back
        frame_f = frame.astype(np.float32)
        color_heatmap_f = color_heatmap.astype(np.float32)
        
        overlay = (frame_f * (1 - alpha_mask_3d) + color_heatmap_f * alpha_mask_3d).astype(np.uint8)
        
        return overlay
