from collections import defaultdict, deque

# ─── Traffic level thresholds ─────────────────────────────────────────────────
# These are measurable visit counts per analysis session.
# Adjust to suit your environment / session length.
TRAFFIC_HIGH_VISITS = 10    # >= 10 visits → HIGH
TRAFFIC_MEDIUM_VISITS = 4   # 4–9 visits  → MEDIUM
                             # < 4 visits   → LOW


class OccupancyAnalyzer:
    """Analyzes occupancy, dwell times, zone transitions, and traffic levels.
    Zones are defined as (x1_pct, y1_pct, x2_pct, y2_pct) – relative to frame size.
    """

    def __init__(self):
        self.reset_occupancy()
        self.zones = self._default_zones()
        self.zone_visits = defaultdict(list)      # zone -> list of dwell frames
        self.active_tracks = {}                   # track_id -> {zone, entry_frame}
        self.track_transitions = defaultdict(list) # track_id -> ordered zone list
        self.frame_width = None
        self.frame_height = None
        self.fps = 30.0

    # ── Occupancy helpers ────────────────────────────────────────────────────
    def reset_occupancy(self):
        self.frame_count = 0
        self.current_occupancy = 0
        self.peak_occupancy = 0
        self.cumulative_occupancy = 0
        self.average_occupancy = 0.0
        self.unique_people = set()
        self.occupancy_history = deque(maxlen=100)

    def reset(self):
        """Reset analytics only (called by /control/reset). Does not clear frame dimensions."""
        self.reset_occupancy()
        self.zone_visits.clear()
        self.active_tracks.clear()
        self.track_transitions.clear()
        # Keep frame_width / frame_height so next frame processes correctly

    # ── Zone configuration ───────────────────────────────────────────────────
    def _default_zones(self):
        """Zones as relative (x1, y1, x2, y2) percentages of frame.
        Exit is checked first since it overlaps with other zones at the top."""
        return {
            "Exit":      (0.00, 0.00, 1.00, 0.15),
            "Entrance":  (0.00, 0.15, 0.25, 1.00),
            "Main Area": (0.25, 0.15, 0.75, 1.00),
            "Counter":   (0.75, 0.15, 1.00, 1.00),
        }

    def set_frame_dimensions(self, width: int, height: int, fps: float = 30.0):
        self.frame_width = width
        self.frame_height = height
        self.fps = fps if fps > 0 else 30.0

    def update_frame_info(self, width: int, height: int, fps: float = 30.0):
        """Called by the processing thread. Always updates dimensions."""
        self.set_frame_dimensions(width, height, fps)

    def _abs_zone_rect(self, name):
        rel = self.zones[name]
        return (
            int(rel[0] * self.frame_width),
            int(rel[1] * self.frame_height),
            int(rel[2] * self.frame_width),
            int(rel[3] * self.frame_height),
        )

    def _point_in_zone(self, cx, cy, zone_name):
        x1, y1, x2, y2 = self._abs_zone_rect(zone_name)
        return x1 <= cx <= x2 and y1 <= cy <= y2

    def _detect_zone(self, cx, cy):
        """Return zone name containing (cx, cy), or None. Returns None if frame dimensions not set."""
        if self.frame_width is None or self.frame_height is None:
            return None
        for name in self.zones:
            if self._point_in_zone(cx, cy, name):
                return name
        return None

    # ── Traffic classification ───────────────────────────────────────────────
    def classify_traffic(self, visits: int) -> str:
        """Classify a zone's traffic level based on total visits."""
        if visits >= TRAFFIC_HIGH_VISITS:
            return "HIGH"
        elif visits >= TRAFFIC_MEDIUM_VISITS:
            return "MEDIUM"
        else:
            return "LOW"

    # ── Main update loop ─────────────────────────────────────────────────────
    def update(self, tracked_objects):
        """Update all analytics for one frame. Returns the full stats dict."""
        self.frame_count += 1
        self.current_occupancy = len(tracked_objects)

        if self.current_occupancy > self.peak_occupancy:
            self.peak_occupancy = self.current_occupancy
        self.cumulative_occupancy += self.current_occupancy
        self.average_occupancy = self.cumulative_occupancy / self.frame_count

        for obj in tracked_objects:
            self.unique_people.add(obj["track_id"])

        # Dwell-time / zone handling
        current_ids = {obj["track_id"] for obj in tracked_objects}

        for obj in tracked_objects:
            tid = obj["track_id"]
            cx, cy = obj["center_x"], obj["center_y"]
            zone = self._detect_zone(cx, cy)

            if tid in self.active_tracks:
                prev_zone = self.active_tracks[tid]["zone"]
                if prev_zone == zone:
                    continue   # still in same zone
                # Exited previous zone
                if prev_zone is not None:
                    dwell = self.frame_count - self.active_tracks[tid]["entry"]
                    self.zone_visits[prev_zone].append(dwell)
                # Entered new zone
                if zone is not None:
                    self.track_transitions[tid].append(zone)
                    self.active_tracks[tid] = {"zone": zone, "entry": self.frame_count}
                else:
                    del self.active_tracks[tid]
            else:
                if zone is not None:
                    self.track_transitions[tid].append(zone)
                    self.active_tracks[tid] = {"zone": zone, "entry": self.frame_count}

        # Tracks that disappeared
        missing_ids = set(self.active_tracks.keys()) - current_ids
        for tid in missing_ids:
            info = self.active_tracks[tid]
            if info["zone"] is not None:
                dwell = self.frame_count - info["entry"]
                self.zone_visits[info["zone"]].append(dwell)
            del self.active_tracks[tid]

        # Keep 100-point rolling history
        self.occupancy_history.append({
            "frame": self.frame_count,
            "occupancy": self.current_occupancy,
        })

        return self.get_stats()

    # ── Stats aggregation ────────────────────────────────────────────────────
    def get_stats(self):
        fps = self.fps or 30.0

        # Occupancy
        occupancy = {
            "current_occupancy": self.current_occupancy,
            "peak_occupancy": self.peak_occupancy,
            "average_occupancy": round(self.average_occupancy, 2),
            "unique_people": len(self.unique_people),
            "history": list(self.occupancy_history),
        }

        # Dwell stats + traffic classification per zone
        dwell_stats = []
        traffic_stats = []
        total_dwell_all = []

        for zone in self.zones:
            times = self.zone_visits.get(zone, [])
            visits = len(times)
            # Convert dwell from frames to seconds
            times_sec = [round(t / fps, 2) for t in times]
            avg_s = round(sum(times_sec) / visits, 2) if visits else 0.0
            min_s = min(times_sec) if times_sec else 0.0
            max_s = max(times_sec) if times_sec else 0.0
            total_s = round(sum(times_sec), 2)

            total_dwell_all.extend(times_sec)
            traffic_level = self.classify_traffic(visits)

            dwell_stats.append({
                "zone": zone,
                "visits": visits,
                "average_dwell_seconds": avg_s,
                "minimum_dwell_seconds": min_s,
                "maximum_dwell_seconds": max_s,
            })

            traffic_stats.append({
                "zone": zone,
                "visits": visits,
                "traffic_level": traffic_level,
                "average_dwell_seconds": avg_s,
            })

        # Global average dwell
        avg_dwell_global = round(sum(total_dwell_all) / len(total_dwell_all), 2) if total_dwell_all else 0.0

        # Zone transitions (track_id -> ordered zones)
        transitions = {
            str(tid): zones
            for tid, zones in self.track_transitions.items()
            if zones
        }

        # Transition matrix (from_zone -> to_zone -> count)
        matrix = defaultdict(lambda: defaultdict(int))
        zone_names = list(self.zones.keys())
        for tid, zones in self.track_transitions.items():
            for i in range(len(zones) - 1):
                matrix[zones[i]][zones[i + 1]] += 1

        matrix_out = {
            from_z: {to_z: matrix[from_z].get(to_z, 0) for to_z in zone_names}
            for from_z in zone_names
        }

        return {
            **occupancy,
            "average_dwell_global": avg_dwell_global,
            "dwell_stats": dwell_stats,
            "traffic_stats": traffic_stats,
            "zone_transitions": transitions,
            "transition_matrix": matrix_out,
            "zone_names": zone_names,
        }
