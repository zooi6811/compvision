import cv2
import numpy as np
from marker_lookup import id_to_grid # Ensure this file is in the same folder
import threading
import time

# -------------------------
# Settings
# -------------------------
# Marker Geometry Settings
GRID_SIZE = 4
BORDER = 1
WARP_SIZE = 200     

# Detection Thresholds
MIN_AREA = 1000
AVG_BORDER_THRESHOLD = 140

# Smoothing Settings (One-Euro Filter)
SMOOTHING_MIN_CUTOFF = 0.1
SMOOTHING_D_CUTOFF = 0.7
SMOOTHING_BETA = 0.007
MARKER_TIMEOUT = 1.5

# Image Preprocessing Settings (CLAHE)
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

# Debug Flag (Controls visual output)
# This can be toggled from main.py
DEBUG = False 

# -------------------------
# One-Euro Filter Classes
# -------------------------
class LowPassFilter:
    def __init__(self):
        self.s = None
    def __call__(self, x, alpha):
        if self.s is None: self.s = x
        else: self.s = alpha * x + (1.0 - alpha) * self.s
        return self.s

class OneEuroFilter:
    def __init__(self, min_cutoff, beta, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_filter = LowPassFilter()
        self.dx_filter = LowPassFilter()
        self.t_prev = None
        self.x_prev = None

    def smoothing_factor(self, dt, cutoff):
        r = 2.0 * np.pi * cutoff * dt
        return r / (r + 1.0)

    def __call__(self, t, x):
        if self.t_prev is None:
            self.t_prev = t
            self.x_prev = x
            return x

        dt = t - self.t_prev
        if dt <= 1e-6: return self.x_prev

        dx = (x - self.x_prev) / dt
        alpha_d = self.smoothing_factor(dt, self.d_cutoff)
        edx = self.dx_filter(dx, alpha_d)

        cutoff = self.min_cutoff + self.beta * np.abs(edx)

        alpha = self.smoothing_factor(dt, cutoff)
        x_filtered = self.x_filter(x, alpha)

        self.t_prev = t
        self.x_prev = x_filtered
        
        return x_filtered

# -------------------------
# Precompute rotated patterns
# -------------------------
grid_hash_map = {}
def _precompute_grids():
    """Initializes the grid_hash_map for fast marker lookup."""
    try:
        for seq_id, grid in id_to_grid.items():
            g = np.array(grid, dtype=np.uint8).reshape(GRID_SIZE, GRID_SIZE)
            for rot in range(4):
                g_rot = np.rot90(g, rot)
                # Store the ID and rotation for each unique rotated pattern
                grid_hash_map[_grid_key_bytes(g_rot)] = (seq_id, rot)
    except Exception as e:
        print(f"WARNING: Could not precompute marker grids. Is marker_lookup.py present? Error: {e}")

def _grid_key_bytes(grid):
    return np.ascontiguousarray(grid.astype(np.uint8)).tobytes()

# Execute precomputation on import
_precompute_grids()

# -------------------------
# Helpers (order_corners, four_point_warp, etc.)
# -------------------------
def order_corners(pts):
    """Orders the 4 detected corners to be Top-Left, Top-Right, Bottom-Right, Bottom-Left."""
    pts = pts.reshape(4, 2)
    centroid = pts.mean(axis=0)
    rect = np.zeros((4, 2), dtype=np.float32)
    
    # Logic to classify corners based on centroid (Simplified version from original)
    for pt in pts:
        is_top = pt[1] < centroid[1]
        is_left = pt[0] < centroid[0]

        if is_top and is_left:
            rect[0] = pt # Top-Left
        elif is_top and not is_left:
            rect[1] = pt # Top-Right
        elif not is_top and not is_left:
            rect[2] = pt # Bottom-Right
        elif not is_top and is_left:
            rect[3] = pt # Bottom-Left
            
    # Fallback for unstable ordering (kept from original)
    if np.any(rect.sum(axis=1) == 0):
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
            
    return rect

def four_point_warp(image, pts, size=WARP_SIZE):
    """Warps a quadrilateral area into a square image."""
    rect = pts
    dst = np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (size, size))

def sample_grid_fast(warped, grid_size=GRID_SIZE, border=BORDER):
    """Downsamples the center of the warped image to get the marker grid."""
    h, w = warped.shape
    full = grid_size + 2 * border
    # Calculate crop area based on border
    top = int(round(border * h / full)); bottom = int(round(h - border * h / full))
    left = int(round(border * w / full)); right = int(round(w - border * w / full))
    inner = warped[top:bottom, left:right] if bottom > top and right > left else warped
    # Resize inner part to the grid size
    inner_small = cv2.resize(inner, (grid_size, grid_size), interpolation=cv2.INTER_AREA)
    # Convert to binary (True/False or 1/0) based on standard threshold
    return (inner_small < 127).astype(np.uint8)

def validate_border_by_average(warped, grid_size=GRID_SIZE, border=BORDER, avg_dark_threshold=AVG_BORDER_THRESHOLD):
    """Checks if the marker's border (the area around the grid) is sufficiently dark."""
    h, w = warped.shape
    full = grid_size + 2 * border
    cell_averages = cv2.resize(warped, (full, full), interpolation=cv2.INTER_AREA)
    # Create a mask for the border cells
    border_mask = np.ones((full, full), dtype=np.uint8); border_mask[border:-border, border:-border] = 0
    border_cell_values = cell_averages[border_mask == 1]
    if border_cell_values.size == 0: return False
    avg_intensity = np.mean(border_cell_values)
    return avg_intensity < avg_dark_threshold

# -------------------------
# Detection pipeline
# -------------------------
def _detect_markers_internal(frame, min_area=MIN_AREA):
    """
    Performs the raw marker detection and decoding on a single frame.
    Returns: A list of raw detected markers and the debug frame.
    """
    frame_debug = frame.copy() if DEBUG else frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE for improved contrast
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE)
    gray_normalized = clahe.apply(gray)
    
    # Adaptive Thresholding (applied to normalized image)
    thr = cv2.adaptiveThreshold(
        gray_normalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 21, 1 
    )

    # Morphological Opening for smoother contours
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) 

    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    found = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > 80000: continue
        
        # Solidity Check
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0: continue
        solidity = float(area) / hull_area
        if solidity < 0.7: continue
        
        # Quadrilateral fitting attempt
        approx = None
        peri = cv2.arcLength(cnt, True)
        for eps in [0.05, 0.07, 0.1]: 
            approx = cv2.approxPolyDP(cnt, eps * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx): break
        else:
            if DEBUG: cv2.drawContours(frame_debug, [hull], -1, (0, 0, 255), 2) # Red Hull on failure
            continue

        raw_corners = approx.astype(np.float32)
        ordered_corners = order_corners(raw_corners)

        # Border Validation: Warp the original gray image
        warped = four_point_warp(gray, ordered_corners) 
        if not validate_border_by_average(warped): continue

        # Grid Sample & Lookup: Warp the normalized image for decoding
        warped_normalized = four_point_warp(gray_normalized, ordered_corners) 
        grid = sample_grid_fast(warped_normalized)
        key = _grid_key_bytes(grid)
        
        if key not in grid_hash_map: continue

        seq_id, rot = grid_hash_map[key]
        
        # If all checks pass
        found.append({"id": seq_id, "corners": ordered_corners, "rotation": rot})
        if DEBUG: cv2.drawContours(frame_debug, [cnt], -1, (255, 255, 255), 1) # White line on success

    return found, frame_debug

# -------------------------
# Reusable Tracker Class
# -------------------------
class MarkerTracker:
    """
    Manages the detection, filtering, and tracking of custom markers.
    """
    def __init__(self, min_cutoff=SMOOTHING_MIN_CUTOFF, d_cutoff=SMOOTHING_D_CUTOFF, 
                 beta=SMOOTHING_BETA, timeout=MARKER_TIMEOUT):
        """Initializes the tracker with smoothing parameters."""
        self.marker_filters = {}
        self.min_cutoff = min_cutoff
        self.d_cutoff = d_cutoff
        self.beta = beta
        self.timeout = timeout
        print("Custom MarkerTracker initialized.")

    def process_frame(self, frame, current_time=None):
        """
        Detects markers, applies One-Euro filter smoothing, and cleans up expired filters.
        Returns:
            tuple: (tracked_markers, debug_frame)
                - tracked_markers (list): [{"id": int, "corners": np.ndarray(4, 2), "center": np.ndarray(2)}]
                - debug_frame (np.ndarray): Frame with debug drawings (if DEBUG=True).
        """
        if current_time is None:
            current_time = time.time()
            
        detected_markers, debug_frame = _detect_markers_internal(frame)
        
        tracked_markers = []
        seen_ids = set()

        for m in detected_markers:
            marker_id = m['id']
            detected_corners = m['corners']
            rotation = m['rotation']
            seen_ids.add(marker_id)
            
            # --- Filter Management ---
            if marker_id not in self.marker_filters:
                filters = [OneEuroFilter(self.min_cutoff, self.beta, d_cutoff=self.d_cutoff) 
                           for _ in range(4)]
                self.marker_filters[marker_id] = {"filters": filters, "last_seen": current_time}
            else:
                self.marker_filters[marker_id]['last_seen'] = current_time

            # --- Apply Filters ---
            filters = self.marker_filters[marker_id]['filters']
            smoothed_corners = np.zeros((4, 2), dtype=np.float32)
            
            for i in range(4):
                smoothed_corners[i] = filters[i](current_time, detected_corners[i])
            
            # --- Prepare Output Data ---
            center = np.mean(smoothed_corners, axis=0)
            
            tracked_markers.append({
                "id": marker_id, 
                "corners": smoothed_corners, # Return smoothed corners
                "center": center.astype(int),
                "rotation": rotation 
            })
            
            # --- DEBUG DRAWING (in the main loop's original style) ---
            if DEBUG:
                self._draw_marker(debug_frame, marker_id, smoothed_corners.astype(int), rotation)

        # --- Garbage Collection ---
        expired_ids = [
            marker_id for marker_id, data in self.marker_filters.items()
            if (current_time - data['last_seen']) > self.timeout
        ]
        
        for marker_id in expired_ids:
            if DEBUG: print(f"Garbage collecting filter for ID {marker_id}")
            del self.marker_filters[marker_id]

        return tracked_markers, debug_frame

    def _draw_marker(self, frame_to_show, marker_id, corners, rotation):
        """Helper to draw successful detections on the debug frame."""
        corner_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)] # R, G, B, C
        
        # Green smoothed quadrilateral outline
        cv2.polylines(frame_to_show, [corners], True, (0, 255, 0), 2)
        
        # Colored canonical corners (R, G, B, C)
        for i, color in enumerate(corner_colors):
            detected_idx = (i - rotation) % 4
            cv2.circle(frame_to_show, tuple(corners[detected_idx]), 6, color, -1)
        
        # Text ID
        text_corner_idx = (0 - rotation) % 4
        cv2.putText(frame_to_show, f"ID:{marker_id}", tuple(corners[text_corner_idx]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Center Position
        center = np.mean(corners, axis=0).astype(int)
        cv2.circle(frame_to_show, tuple(center), 3, (255, 255, 255), -1)