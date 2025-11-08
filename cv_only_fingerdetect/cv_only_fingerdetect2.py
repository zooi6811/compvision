# Tried adding optical flow to improve tracking

import cv2
import numpy as np
from collections import deque
import time

# -------------------- PARAMETERS --------------------
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

SKIN_HSV_LOWER = np.array([1, 70, 155])
SKIN_HSV_UPPER = np.array([12, 125, 254])

MIN_CONTOUR_AREA = 500
SMOOTHING_WINDOW = 5
DRAW_COLOR = (0, 0, 255)
THICKNESS = 4
ALPHA = 0.8

lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

# -------------------- MODULAR FUNCTIONS --------------------
def setup_camera(width=FRAME_WIDTH, height=FRAME_HEIGHT):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def get_skin_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, SKIN_HSV_LOWER, SKIN_HSV_UPPER)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    return mask


def select_hand_contour(contours, frame_height):
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_CONTOUR_AREA or area > frame_height*frame_height/2:
            continue
        M = cv2.moments(cnt)
        if M['m00'] == 0:
            continue
        cy = int(M['m01']/M['m00'])
        if cy < frame_height // 2:
            continue
        candidates.append(cnt)
    if not candidates:
        return None
    return max(candidates, key=cv2.contourArea)


def find_fingertip(contour):
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
        if depth / 256.0 < 20:
            continue
        candidates.append(tuple(contour[s][0]))
        candidates.append(tuple(contour[e][0]))
    if candidates:
        return min(candidates, key=lambda p: p[1])
    return tuple(contour[contour[:, :, 1].argmin()][0])


def smooth_points(pts):
    if not pts:
        return None
    avg_x = int(sum([p[0] for p in pts]) / len(pts))
    avg_y = int(sum([p[1] for p in pts]) / len(pts))
    return (avg_x, avg_y)


def draw_on_canvas(canvas, prev, curr):
    if prev is not None and curr is not None:
        cv2.line(canvas, prev, curr, DRAW_COLOR, THICKNESS)


def overlay_canvas(frame, canvas):
    return cv2.addWeighted(frame, 1.0, canvas, ALPHA, 0)


def track_fingertip_with_LK(prev_gray, curr_gray, p0):
    """Track fingertip using Lucas–Kanade optical flow."""
    if prev_gray is None or p0 is None:
        return None, None, False
    p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None, **lk_params)
    if p1 is None or st[0][0] == 0:
        return None, None, False
    new_point = tuple(p1[0, 0].astype(int))
    return p1, new_point, True


# -------------------- MAIN FUNCTION --------------------

def main():
    cap = setup_camera()
    canvas = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    pts = deque(maxlen=SMOOTHING_WINDOW)
    old_gray = None
    p0 = None
    tracking_status = "none"
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect fingertip with previous method (when not tracking) 
        mask = get_skin_mask(frame)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hand_contour = select_hand_contour(contours, FRAME_HEIGHT)

        fingertip = None
        if hand_contour is not None and p0 is None:
            fingertip = find_fingertip(hand_contour)
            if fingertip is not None:
                pts.appendleft(fingertip)
                cv2.circle(frame, fingertip, 8, (255, 0, 0), -1)
                p0 = np.array([[fingertip]], dtype=np.float32)
                old_gray = gray.copy()
                tracking_status = "detect"

        # If already tracking, update using LK optical flow
        elif p0 is not None:
            p1, new_point, ok = track_fingertip_with_LK(old_gray, gray, p0)
            if ok:
                pts.appendleft(new_point)
                cv2.circle(frame, new_point, 6, (0, 255, 255), -1)
                p0 = p1
                tracking_status = "track"
            else:
                p0 = None
                tracking_status = "lost"

        # Drawing 
        smoothed = smooth_points(pts)
        if smoothed and len(pts) >= 2:
            draw_on_canvas(canvas, pts[1], smoothed)

        overlay = overlay_canvas(frame, canvas)
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time)
        prev_time = curr_time

        cv2.putText(overlay, f"Status: {tracking_status}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.putText(overlay, f"FPS: {fps:.1f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow('mask', mask)
        cv2.imshow('draw', overlay)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            canvas[:] = 0
            pts.clear()
        elif key == ord('r'):
            p0 = None
            pts.clear()
            tracking_status = "none"

        old_gray = gray.copy()

    cap.release()
    cv2.destroyAllWindows()



# Separate function that helps to find and tune skin colour thresholds
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


    

if __name__ == "__main__":
    main()
    # color_picker()

