# Attempt at making a solid finger-tracking solution using only CV
# Uses contours + convex features to determine where the drawing finger (sticking up on a hand) is. 

# Problem with this one is that there are times when the face is inside the frame and intersects with the hand, it might become part of the detected hand. 


import cv2
import numpy as np
from collections import deque
import time


FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# thresholds based on personal experimentation
SKIN_HSV_LOWER = np.array([1, 70, 155])
SKIN_HSV_UPPER = np.array([12, 125, 254])

MIN_CONTOUR_AREA = 500
SMOOTHING_WINDOW = 5
DRAW_COLOR = (0, 0, 255)
THICKNESS = 4
ALPHA = 0.8

# Optical flow params
lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

frame_for_hsv = None


def get_hand_centroid(contour):
    M = cv2.moments(contour)
    if M['m00'] == 0:
        return None
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    return (cx, cy)
    

# -------------------- UTILITY FUNCTIONS --------------------
def setup_camera(width=FRAME_WIDTH, height=FRAME_HEIGHT):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def get_skin_mask(frame):
    """Return a binary mask where skin pixels are white."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, SKIN_HSV_LOWER, SKIN_HSV_UPPER)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    return mask


def find_fingertip(contour):
    """Detect fingertip from hand contour."""
    if contour is None or len(contour) < 5:
        return None
    hull = cv2.convexHull(contour, returnPoints=False)
    if hull is None or len(hull) < 3:
        return None
    defects = cv2.convexityDefects(contour, hull)
    if defects is None:
        return tuple(contour[contour[:, :, 1].argmin()][0])

    candidates = []
    for i in range(defects.shape[0]):
        s, e, f, depth = defects[i, 0]
        depth = depth / 256.0
        if depth < 20:
            continue
        candidates.append(tuple(contour[s][0]))
        candidates.append(tuple(contour[e][0]))
    if candidates:
        return min(candidates, key=lambda p: p[1])
    return tuple(contour[contour[:, :, 1].argmin()][0])


def draw_on_canvas(canvas, prev, curr):
    if prev is not None and curr is not None:
        cv2.line(canvas, prev, curr, DRAW_COLOR, THICKNESS)


def overlay_canvas(frame, canvas):
    return cv2.addWeighted(frame, 1.0, canvas, ALPHA, 0)


def show_hsv(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and frame_for_hsv is not None:
        hsv = cv2.cvtColor(frame_for_hsv, cv2.COLOR_BGR2HSV)
        print(f"HSV at ({x},{y}): {hsv[y, x]}")

def smooth_points(pts):
    """Return averaged coordinates from deque of points for smoothing."""
    if not pts:
        return None
    avg_x = int(sum([p[0] for p in pts]) / len(pts))
    avg_y = int(sum([p[1] for p in pts]) / len(pts))
    return (avg_x, avg_y)

def select_hand_contour(contours, frame_height):
    """Return the contour most likely to be the hand."""
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_CONTOUR_AREA or area > frame_height*frame_height/2:
            continue
        # Optionally, discard contours whose centroid is in upper frame (face)
        M = cv2.moments(cnt)
        if M['m00'] == 0:
            continue
        cy = int(M['m01']/M['m00'])
        if cy < frame_height//2:
            continue
        candidates.append(cnt)
    if not candidates:
        return None
    
    # pick largest among remaining
    candidates = sorted(candidates, key=cv2.contourArea, reverse=True)
    return candidates[0]



def clear_canvas_and_tracking(canvas, pts):
    """Clear canvas, tracked points, and reset fingertip tracking."""
    canvas[:] = 0
    pts.clear()
    return None  # This can reset p0 in main loop



def main():
    cap = setup_camera()
    canvas = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    pts = deque(maxlen=SMOOTHING_WINDOW)

    cv2.namedWindow('draw')
    cv2.setMouseCallback('draw', show_hsv)

    old_gray = None
    p0 = None  # fingertip for tracking
    tracking_status = "none"  # none / detect / track / lost
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        global frame_for_hsv
        frame_for_hsv = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # --- Skin detection ---
        mask = get_skin_mask(frame)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # --- Select hand contour robustly ---
        hand_contour = select_hand_contour(contours, FRAME_HEIGHT)

        if hand_contour is not None:
            # Draw hand contour (green) and convex hull (red)
            cv2.drawContours(frame, [hand_contour], -1, (0, 255, 0), 2)
            hull = cv2.convexHull(hand_contour)
            cv2.drawContours(frame, [hull], -1, (0, 0, 255), 2)

            # Uses convex hull to determine which finger is sticking up the most (i.e., the draw finger)

            # Fingertip detection
            fingertip = find_fingertip(hand_contour)
            if fingertip is not None:
                pts.appendleft(fingertip)
                cv2.circle(frame, fingertip, 8, (255, 0, 0), -1)  # raw fingertip
                p0 = np.array([[fingertip]], dtype=np.float32)
                old_gray = gray.copy()
                tracking_status = "detect"
        else:
            tracking_status = "none"
            p0 = None

        # --- Smooth fingertip for drawing ---
        smoothed = smooth_points(pts)
        if smoothed and len(pts) >= 2:
            draw_on_canvas(canvas, pts[1], smoothed)
            cv2.circle(frame, smoothed, 5, (0, 255, 255), -1)  # smoothed point

        # --- Overlay and FPS ---
        overlay = overlay_canvas(frame, canvas)
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time)
        prev_time = curr_time

        # Draw status
        color = (200, 200, 200)
        text = f"Status: {tracking_status}"
        cv2.putText(overlay, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(overlay, f"FPS: {fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

        # --- Display ---
        cv2.imshow('mask', mask)
        cv2.imshow('draw', overlay)

        # --- Controls ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            canvas[:] = 0
            pts.clear()
        elif key == ord('r'):
            p0 = None
            pts.clear()

        # elif key == ord('c'):
        #     clear_canvas(canvas, pts)

        # elif key == ord('r'):
        #     p0 = clear_canvas_and_tracking(canvas, pts)


    cap.release()
    cv2.destroyAllWindows()


# Function helps to find and tune skin colour thresholds
def color_picker():
    cap = cv2.VideoCapture(0)

    def show_values(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            frame = param
            bgr = frame[y, x]
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[y, x]
            ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)[y, x]
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)[y, x]
            print(f"Pixel at ({x},{y}):")
            print(f" BGR  = {bgr}")
            print(f" HSV  = {hsv}")
            print(f" YCrCb= {ycrcb}")
            print(f" LAB  = {lab}\n")

    cv2.namedWindow('Color Picker')

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        cv2.setMouseCallback('Color Picker', show_values, param=frame)
        cv2.imshow('Color Picker', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# -------------------- ENTRY POINT --------------------
if __name__ == "__main__":
    main()
    # color_picker()

