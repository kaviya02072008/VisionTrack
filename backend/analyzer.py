import math
from collections import defaultdict, deque

# ─── Traffic level thresholds ─────────────────────────────────────────────────
# These are measurable visit counts per analysis session.
# Adjust to suit your environment / session length.
TRAFFIC_HIGH_VISITS = 10    # >= 10 visits → HIGH
TRAFFIC_MEDIUM_VISITS = 4   # 4–9 visits  → MEDIUM
                             # < 4 visits   → LOW


class IdentityRegistry:
    """Maintains confirmed unique person identities across an analysis session.
    
    Prevents duplicate unique counting by:
    1. Confirming identities only after appearance in multiple frames or high-confidence tracking.
    2. Associating new or fragmented ByteTrack IDs with recently inactive identities within spatial tolerance.
    3. Strictly guaranteeing that two detections in the same frame NEVER share an identity.
    4. Accumulating continuous dwell times and trajectories across short detection gaps.
    """

    def __init__(self, confirmation_threshold=2, max_inactive_frames=90, reid_dist_pct=0.07):
        self.confirmation_threshold = confirmation_threshold
        self.max_inactive_frames = max_inactive_frames  # ~10s at 9 fps
        self.reid_dist_pct = reid_dist_pct              # % of min(width, height)
        
        self.identities = {}          # canonical_id -> dict of identity state
        self.track_to_identity = {}   # track_id (str) -> canonical_id
        self.track_appearances = defaultdict(int)
        self.next_identity_num = 1
        self.total_detections = 0

    def reset(self):
        self.identities.clear()
        self.track_to_identity.clear()
        self.track_appearances.clear()
        self.next_identity_num = 1
        self.total_detections = 0

    def update(self, tracked_objects, frame_idx, frame_w, frame_h, fps=30.0):
        """
        Processes tracked objects for the current frame.
        Returns:
            active_canonical_objects: list of tracked objects augmented with 'canonical_id'
        """
        self.total_detections += len(tracked_objects)

        if frame_w and frame_h:
            max_reid_dist = max(60.0, min(frame_w, frame_h) * self.reid_dist_pct)
        else:
            max_reid_dist = 80.0

        claimed_in_frame = set()
        active_canonical_objects = []

        # Pass 1: Objects whose track_id is already mapped to a canonical identity
        unassigned_objects = []
        for obj in tracked_objects:
            tid = str(obj["track_id"])
            self.track_appearances[tid] += 1
            cid = self.track_to_identity.get(tid)

            if cid is not None and cid in self.identities and cid not in claimed_in_frame:
                claimed_in_frame.add(cid)
                ident = self.identities[cid]
                ident["last_seen_frame"] = frame_idx
                ident["last_center"] = (obj["center_x"], obj["center_y"])
                ident["last_box"] = (obj.get("x1"), obj.get("y1"), obj.get("x2"), obj.get("y2"))
                ident["total_frames"] += 1
                ident["active_track_id"] = tid

                aug_obj = dict(obj)
                aug_obj["canonical_id"] = cid
                active_canonical_objects.append(aug_obj)
            else:
                unassigned_objects.append(obj)

        # Pass 2: Spatial re-association for unassigned objects against recently lost inactive identities
        # We only consider identities NOT claimed in this frame and seen within max_inactive_frames
        inactive_identities = [
            cid for cid, ident in self.identities.items()
            if cid not in claimed_in_frame and (frame_idx - ident["last_seen_frame"]) <= self.max_inactive_frames
        ]

        still_unassigned = []
        for obj in unassigned_objects:
            tid = str(obj["track_id"])
            cx, cy = obj["center_x"], obj["center_y"]
            bw = (obj.get("x2", cx + 20) - obj.get("x1", cx - 20))
            bh = (obj.get("y2", cy + 40) - obj.get("y1", cy - 40))
            new_area = max(1.0, float(bw * bh))

            best_cid = None
            min_dist = float("inf")

            for cid in inactive_identities:
                if cid in claimed_in_frame:
                    continue
                ident = self.identities[cid]
                lx, ly = ident["last_center"]
                dist = math.hypot(cx - lx, cy - ly)

                # Check bounding box scale compatibility if box exists
                if ident.get("last_box") and all(v is not None for v in ident["last_box"]):
                    obx1, oby1, obx2, oby2 = ident["last_box"]
                    old_area = max(1.0, float((obx2 - obx1) * (oby2 - oby1)))
                    ratio = new_area / old_area
                    if ratio < 0.35 or ratio > 2.8:
                        continue  # vastly different box sizes

                if dist < max_reid_dist and dist < min_dist:
                    min_dist = dist
                    best_cid = cid

            if best_cid is not None:
                claimed_in_frame.add(best_cid)
                self.track_to_identity[tid] = best_cid
                ident = self.identities[best_cid]
                ident["last_seen_frame"] = frame_idx
                ident["last_center"] = (cx, cy)
                ident["last_box"] = (obj.get("x1"), obj.get("y1"), obj.get("x2"), obj.get("y2"))
                ident["total_frames"] += 1
                ident["active_track_id"] = tid

                aug_obj = dict(obj)
                aug_obj["canonical_id"] = best_cid
                active_canonical_objects.append(aug_obj)
            else:
                still_unassigned.append(obj)

        # Pass 3: Confirm new identities for persistent tracks
        for obj in still_unassigned:
            tid = str(obj["track_id"])
            conf = obj.get("confidence", 0.0)
            apps = self.track_appearances[tid]

            # Require confirmation threshold (or high confidence track from ByteTrack)
            is_confirmed = (apps >= self.confirmation_threshold) or (not tid.startswith("u") and conf >= 0.35)
            if is_confirmed:
                cid = f"P{self.next_identity_num}"
                self.next_identity_num += 1
                self.track_to_identity[tid] = cid
                claimed_in_frame.add(cid)

                self.identities[cid] = {
                    "id": cid,
                    "first_seen_frame": frame_idx,
                    "last_seen_frame": frame_idx,
                    "total_frames": 1,
                    "last_center": (obj["center_x"], obj["center_y"]),
                    "last_box": (obj.get("x1"), obj.get("y1"), obj.get("x2"), obj.get("y2")),
                    "active_track_id": tid,
                }
                aug_obj = dict(obj)
                aug_obj["canonical_id"] = cid
                active_canonical_objects.append(aug_obj)
            else:
                # Tentative / unconfirmed (e.g. 1st frame observation)
                aug_obj = dict(obj)
                aug_obj["canonical_id"] = f"tentative_{tid}"
                active_canonical_objects.append(aug_obj)

        return active_canonical_objects

    @property
    def confirmed_unique_count(self):
        return len(self.identities)


class OccupancyAnalyzer:
    """Analyzes occupancy, dwell times, zone transitions, and traffic levels.
    Zones are defined as (x1_pct, y1_pct, x2_pct, y2_pct) – relative to frame size.
    """

    def __init__(self):
        self.identity_registry = IdentityRegistry()
        self.reset_occupancy()
        self.zones = self._default_zones()
        self.zone_visits = defaultdict(list)      # zone -> list of dwell frames
        self.active_tracks = {}                   # canonical_id -> {zone, entry_frame}
        self.track_transitions = defaultdict(list) # canonical_id -> ordered zone list
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
        self.total_detections = 0
        self.unique_people = set()
        self.track_appearances = defaultdict(int)
        self.occupancy_history = deque(maxlen=100)
        if hasattr(self, "identity_registry"):
            self.identity_registry.reset()

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
        # Metric A: Current occupancy — exactly the number of people currently visible in this frame
        self.current_occupancy = len(tracked_objects)

        if self.current_occupancy > self.peak_occupancy:
            self.peak_occupancy = self.current_occupancy
        self.cumulative_occupancy += self.current_occupancy
        self.average_occupancy = self.cumulative_occupancy / self.frame_count

        # Map tracked objects to persistent canonical identities
        canonical_objects = self.identity_registry.update(
            tracked_objects,
            frame_idx=self.frame_count,
            frame_w=self.frame_width,
            frame_h=self.frame_height,
            fps=self.fps,
        )

        # Sync unique_people set and total_detections for backwards compatibility and debug metric C
        self.unique_people = set(self.identity_registry.identities.keys())
        self.total_detections = self.identity_registry.total_detections

        # Dwell-time and zone transitions handled by canonical identity
        current_cids = {obj["canonical_id"] for obj in canonical_objects}

        for obj in canonical_objects:
            cid = obj["canonical_id"]
            cx, cy = obj["center_x"], obj["center_y"]
            zone = self._detect_zone(cx, cy)

            if cid in self.active_tracks:
                prev_zone = self.active_tracks[cid]["zone"]
                if prev_zone == zone:
                    continue   # still in same zone
                # Exited previous zone
                if prev_zone is not None:
                    dwell = self.frame_count - self.active_tracks[cid]["entry"]
                    self.zone_visits[prev_zone].append(dwell)
                # Entered new zone
                if zone is not None:
                    self.track_transitions[cid].append(zone)
                    self.active_tracks[cid] = {"zone": zone, "entry": self.frame_count}
                else:
                    del self.active_tracks[cid]
            else:
                if zone is not None:
                    self.track_transitions[cid].append(zone)
                    self.active_tracks[cid] = {"zone": zone, "entry": self.frame_count}

        # Tracks that disappeared
        missing_cids = set(self.active_tracks.keys()) - current_cids
        for cid in missing_cids:
            info = self.active_tracks[cid]
            if info["zone"] is not None:
                dwell = self.frame_count - info["entry"]
                self.zone_visits[info["zone"]].append(dwell)
            del self.active_tracks[cid]

        # Keep 100-point rolling history
        self.occupancy_history.append({
            "frame": self.frame_count,
            "occupancy": self.current_occupancy,
        })

        return self.get_stats()

    # ── Stats aggregation ────────────────────────────────────────────────────
    def get_stats(self):
        fps = self.fps or 30.0

        # Occupancy metrics cleanly separated
        occupancy = {
            "current_occupancy": self.current_occupancy,
            "peak_occupancy": self.peak_occupancy,
            "average_occupancy": round(self.average_occupancy, 2),
            "unique_people": len(self.unique_people),
            "total_detections": self.total_detections,
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
