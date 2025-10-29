# Final CV-Only Hand Tracker
# This version implements the user's idea of a two-stage process:
# 1. CALIBRATION: Hold an open hand to 'sample' the skin tone.
# 2. TRACKING: Use the sampled data to build a robust tracker.
#
# This uses Histogram Back-Projection, which is far more robust
# to lighting than a simple HSV 'inRange' call.

import cv2
import numpy as np
from collections import deque
import math

# -------------------- Global PARAMETERS --------------------
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

MIN_CONTOUR_AREA = 3000  # Area to detect hand for calibration/tracking
SMOOTHING_WINDOW = 5
DRAW_COLOR = (0, 0, 255)
THICKNESS = 4
ALPHA = 0.8

# --- Calibration ---
# We still need a "first guess" filter to find the hand for calibration
# This is only used for the first few seconds.
CALIBRATION_HSV_LOWER = np.array([0, 40, 80])
CALIBRATION_HSV_UPPER = np.array([25, 255, 255])
CALIBRATION_ROI_START = (220, 140)
CALIBRATION_ROI_END = (420, 340)

calibrated = False
hand_histogram = None

# -------------------- UTILITY FUNCTIONS --------------------
def setup_camera(width=FRAME_WIDTH, height=FRAME_HEIGHT):
    """Initialises and configures the camera."""
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap

def draw_on_canvas(canvas, prev, curr):
    """Draws a line on the canvas."""
    if prev is not None and curr is not None:
        cv2.line(canvas, prev, curr, DRAW_COLOR, THICKNESS)

def overlay_canvas(frame, canvas):
    """Overlays the canvas onto the main frame."""
    return cv2.addWeighted(frame, 1.0, canvas, ALPHA, 0)

# -------------------- NEW CALIBRATION & TRACKING LOGIC --------------------

def calibrate_hand_histogram(frame):
    """
    Finds the hand in the calibration ROI, samples the palm,
    and returns an H-S (Hue-Saturation) histogram.
    """
    global hand_histogram, calibrated
    
    # --- 1. Isolate the ROI ---
    roi = frame[CALIBRATION_ROI_START[1]:CALIBRATION_ROI_END[1],
                CALIBRATION_ROI_START[0]:CALIBRATION_ROI_END[0]]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # --- 2. Use the "first guess" filter ---
    mask = cv2.inRange(hsv_roi, CALIBRATION_HSV_LOWER, CALIBRATION_HSV_UPPER)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # --- 3. Find the hand contour ---
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 1000: # Hand must be of decent size
            # --- 4. Find palm and sample a 30x30 patch ---
            (x, y), radius = cv2.minEnclosingCircle(largest_contour)
            palm_center = (int(x), int(y))
            
            # Define sample patch around the palm (clipping to ROI bounds)
            sample_y_start = max(0, palm_center[1] - 15)
            sample_y_end = min(hsv_roi.shape[0], palm_center[1] + 15)
            sample_x_start = max(0, palm_center[0] - 15)
            sample_x_end = min(hsv_roi.shape[1], palm_center[0] + 15)
            
            palm_patch = hsv_roi[sample_y_start:sample_y_end, sample_x_start:sample_x_end]
            
            # Draw the sampled patch for user feedback
            cv2.rectangle(roi, (sample_x_start, sample_y_start), (sample_x_end, sample_y_end), (0, 255, 0), 2)
            
            # --- 5. Calculate the H-S Histogram ---
            # We use Hue (0) and Saturation (1)
            # We ignore Value (V) because it's too sensitive to lighting/shadows
            hist = cv2.calcHist(
                [palm_patch],       # Source
                [0, 1],             # Channels (H and S)
                None,               # Mask
                [18, 25],           # Bins (18 for H, 25 for S)
                [0, 180, 0, 256]    # Ranges (H is 0-180 in OpenCV)
            )
            
            # Normalize the histogram
            hand_histogram = cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
            calibrated = True
            print("Calibration complete! Tracking enabled.")
            return True
            
    return False

def get_skin_mask_backproject(frame, hist):
    """
    Uses the calibrated histogram to perform back-projection,
    creating a high-quality, adaptive skin mask.
    """
    # --- 1. Convert frame to HSV ---
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # --- 2. Calculate Back-Projection ---
    # This creates a "probability map"
    prob_map = cv2.calcBackProject(
        [hsv_frame],        # Source frame
        [0, 1],             # Channels (H and S)
        hist,               # The calibrated histogram
        [0, 180, 0, 256],    # Ranges
        1                   # Scale
    )

    # --- 3. Clean up the map ---
    # Convolve with an ellipse to merge nearby blobs
    kernel_disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    cv2.filter2D(prob_map, -1, kernel_disc, prob_map)
    
    # --- 4. Threshold to get a binary mask ---
    _, mask = cv2.threshold(prob_map, 60, 255, cv2.THRESH_BINARY)
    
    # --- 5. Final morphology cleanup ---
    kernel_morph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_morph, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_morph, iterations=1)
    
    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    return mask

# -------------------- LANDMARK FINDER (Unchanged) --------------------
def find_hand_landmarks(contour, frame_to_draw_on):
    """
    Finds the palm centre and fingertips using Convex Hull and Defects.
    """
    palm_center = None
    fingertips = []

    (x, y), radius = cv2.minEnclosingCircle(contour)
    palm_center = (int(x), int(y))
    cv2.circle(frame_to_draw_on, palm_center, int(radius), (255, 0, 0), 2)
    cv2.circle(frame_to_draw_on, palm_center, 5, (255, 0, 0), -1)

    hull_indices = cv2.convexHull(contour, returnPoints=False)
    
    if len(hull_indices) > 3:
        defects = cv2.convexityDefects(contour, hull_indices)
        if defects is not None:
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start_pt = tuple(contour[s][0])
                end_pt = tuple(contour[e][0])
                far_pt = tuple(contour[f][0])
                
                a = math.dist(start_pt, end_pt)
                b = math.dist(start_pt, far_pt)
                c = math.dist(end_pt, far_pt)
                
                angle_rad = math.acos((b**2 + c**2 - a**2) / (2 * b * c))
                angle_deg = math.degrees(angle_rad)
                dist_from_palm = math.dist(palm_center, start_pt)

                if angle_deg < 90 and dist_from_palm > radius * 0.5:
                    fingertips.append(start_pt)
                    cv2.circle(frame_to_draw_on, far_pt, 5, (0, 255, 0), -1)

    unique_fingertips = list(set(fingertips))
    return palm_center, unique_fingertips

# -------------------- MAIN LOOP --------------------
def main():
    global calibrated, hand_histogram
    
    cap = setup_camera()
    canvas = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    draw_points = deque(maxlen=SMOOTHING_WINDOW)
    cv2.namedWindow('draw')
    
    status = "Calibrating..."
    mask = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        frame_with_debug = frame.copy()

        if not calibrated:
            # --- CALIBRATION MODE ---
            cv2.rectangle(frame_with_debug, CALIBRATION_ROI_START, CALIBRATION_ROI_END, (255, 255, 0), 2)
            cv2.putText(frame_with_debug, "Place OPEN PALM in box", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame_with_debug, "Press 'c' to capture", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                if calibrate_hand_histogram(frame):
                    status = "Calibrated! Tracking..."
                else:
                    status = "Calibration Failed. Try again."
            
        else:
            # --- TRACKING MODE ---
            mask = get_skin_mask_backproject(frame, hand_histogram)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            fingertips = []
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest_contour) > MIN_CONTOUR_AREA:
                    palm_center, fingertips = find_hand_landmarks(largest_contour, frame_with_debug)
                    for tip in fingertips:
                        cv2.circle(frame_with_debug, tip, 8, (0, 255, 255), -1)
                    status = f"Tracking {len(fingertips)} points"
                else:
                    status = "Looking for hand..."
            else:
                status = "Looking for hand..."

            # --- Drawing Logic ---
            if fingertips:
                drawing_tip = min(fingertips, key=lambda p: p[1]) # Use highest point
                draw_points.appendleft(drawing_tip)
            else:
                draw_points.clear()

            if len(draw_points) >= 2:
                draw_on_canvas(canvas, draw_points[1], draw_points[0])

        # --- Display ---
        overlay = overlay_canvas(frame_with_debug, canvas)
        if not calibrated:
            cv2.putText(overlay, status, (CALIBRATION_ROI_START[0], CALIBRATION_ROI_END[1] + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(overlay, f"Status: {status}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            cv2.putText(overlay, "Press 'r' to re-calibrate", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        if mask is not None:
             cv2.imshow('mask', mask)
        cv2.imshow('draw', overlay)

        # --- Controls ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c') and not calibrated: # 'c' for calibrate
            if calibrate_hand_histogram(frame):
                status = "Calibrated! Tracking..."
            else:
                status = "Calibration Failed. Try again."
        elif key == ord('r'): # 'r' for reset
            calibrated = False
            hand_histogram = None
            status = "Calibrating..."
            canvas[:] = 0
            draw_points.clear()
            if 'mask' in locals() or 'mask' in globals():
                cv2.destroyWindow('mask')
            mask = None

    cap.release()
    cv2.destroyAllWindows()

# -------------------- ENTRY POINT --------------------
if __name__ == "__main__":
    main()