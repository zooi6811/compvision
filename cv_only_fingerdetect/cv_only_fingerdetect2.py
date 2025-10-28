# Not too robust Attempt for finger detection & drawing using only traditional CV methods (thresholding, contour finding, optical flow...)
# Forms part of the the discussion for why we wanted to move on to media pipe and ML stuff

import cv2
import numpy as np
from collections import deque

# -------------------- Global PARAMETERS --------------------
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Bounds based on testing
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

# ROI box (calibration)
ROI_START = (250, 150)
ROI_END = (390, 290)
roi_visible = True
calibrated = False
p0 = None  # fingertip point


# -------------------- UTILITY FUNCTIONS --------------------
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


def draw_on_canvas(canvas, prev, curr):
    if prev is not None and curr is not None:
        cv2.line(canvas, prev, curr, DRAW_COLOR, THICKNESS)


def overlay_canvas(frame, canvas):
    return cv2.addWeighted(frame, 1.0, canvas, ALPHA, 0)


def smooth_points(pts):
    if not pts:
        return None
    avg_x = int(sum([p[0] for p in pts]) / len(pts))
    avg_y = int(sum([p[1] for p in pts]) / len(pts))
    return (avg_x, avg_y)


# -------------------- MOUSE CALLBACK -------------------- (had to use this to calibrate for finger)
def select_finger(event, x, y, flags, param):
    global p0, calibrated, roi_visible
    if event == cv2.EVENT_LBUTTONDOWN and roi_visible:
        # If click inside ROI, set fingertip
        if ROI_START[0] <= x <= ROI_END[0] and ROI_START[1] <= y <= ROI_END[1]:
            p0 = np.array([[[x, y]]], dtype=np.float32)
            calibrated = True
            roi_visible = False
            print(f"Calibrated at {x}, {y}")


# -------------------- MAIN LOOP --------------------
def main():
    global frame_for_hsv, p0, calibrated, roi_visible

    cap = setup_camera()
    canvas = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    pts = deque(maxlen=SMOOTHING_WINDOW)

    cv2.namedWindow('draw')
    cv2.setMouseCallback('draw', select_finger)

    old_gray = None
    tracking_status = "idle"

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        frame_for_hsv = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # --- Skin mask & contours ---
        mask = get_skin_mask(frame)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        fingertip_detected = False
        if calibrated and p0 is not None:
            # Optical flow tracking
            if old_gray is not None:
                p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, gray, p0, None, **lk_params)
                if st is not None and st[0][0] == 1:
                    new_pt = tuple(map(int, p1[0][0]))
                    cv2.circle(frame, new_pt, 8, (0, 255, 0), -1)
                    pts.appendleft(new_pt)
                    p0 = np.array([[new_pt]], dtype=np.float32)
                    tracking_status = "tracking"
                    fingertip_detected = True
                else:
                    tracking_status = "lost"
                    fingertip_detected = False
                    continue

            old_gray = gray.copy()

        else:
            tracking_status = "calibrate" if roi_visible else "idle"

        # --- Draw smoothed points ---
        smoothed = smooth_points(pts)
        if smoothed and len(pts) >= 2:
            draw_on_canvas(canvas, pts[1], smoothed)
            cv2.circle(frame, smoothed, 5, (0, 255, 255), -1)

        # --- Overlay ---
        overlay = overlay_canvas(frame, canvas)

        # Draw ROI box if visible
        if roi_visible:
            cv2.rectangle(overlay, ROI_START, ROI_END, (255, 255, 0), 2)
            cv2.putText(overlay, "Place finger and click inside box", (10, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Draw status
        cv2.putText(overlay, f"Status: {tracking_status}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

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
            calibrated = False
            roi_visible = True
            p0 = None
        elif key == ord('r'):
            pts.clear()
            p0 = None
            calibrated = False
            roi_visible = True

    cap.release()
    cv2.destroyAllWindows()


# -------------------- ENTRY POINT --------------------
if __name__ == "__main__":
    main()





# Earlier version commented out here; less robust tracking but doesn't crash easily like the newer one

# import cv2
# import numpy as np
# from collections import deque
# import time

# FRAME_WIDTH = 640
# FRAME_HEIGHT = 480

# MIN_CONTOUR_AREA = 500
# SMOOTHING_WINDOW = 5
# DRAW_COLOR = (0, 0, 255)
# THICKNESS = 4
# ALPHA = 0.8

# # Optical flow parameters
# lk_params = dict(
#     winSize=(15, 15),
#     maxLevel=2,
#     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
# )

# # Sampling box (fixed)
# sampling_box = (FRAME_WIDTH//2 - 50, FRAME_HEIGHT//2 - 50, 100, 100)  # x, y, w, h
# init_point = None
# frame_for_hsv = None

# def setup_camera(width=FRAME_WIDTH, height=FRAME_HEIGHT):
#     cap = cv2.VideoCapture(0)
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
#     return cap

# def draw_on_canvas(canvas, prev, curr):
#     if prev is not None and curr is not None:
#         cv2.line(canvas, prev, curr, DRAW_COLOR, THICKNESS)

# def overlay_canvas(frame, canvas):
#     return cv2.addWeighted(frame, 1.0, canvas, ALPHA, 0)

# def smooth_points(pts):
#     if not pts:
#         return None
#     avg_x = int(sum([p[0] for p in pts]) / len(pts))
#     avg_y = int(sum([p[1] for p in pts]) / len(pts))
#     return (avg_x, avg_y)

# def mouse_callback(event, x, y, flags, param):
#     global init_point
#     if event == cv2.EVENT_LBUTTONDOWN:
#         sx, sy, w, h = sampling_box
#         if sx <= x <= sx+w and sy <= y <= sy+h:
#             init_point = (x, y)
#             print(f"Initial tracking point set at {init_point}")


# def main():
#     global frame_for_hsv, init_point
#     cap = setup_camera()
#     canvas = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
#     pts = deque(maxlen=SMOOTHING_WINDOW)

#     cv2.namedWindow('draw')
#     cv2.setMouseCallback('draw', mouse_callback)

#     old_gray = None
#     p0 = None
#     prev_time = time.time()

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frame = cv2.flip(frame, 1)
#         frame_for_hsv = frame.copy()
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#         # --- Draw sampling box ---
#         sx, sy, w, h = sampling_box
#         cv2.rectangle(frame, (sx, sy), (sx+w, sy+h), (0, 255, 255), 2)
#         cv2.putText(frame, "Click inside box to track finger", (sx, sy-10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

#         # --- Initialize tracking ---
#         if init_point is not None and p0 is None:
#             p0 = np.array([[init_point]], dtype=np.float32)
#             old_gray = gray.copy()

#         # --- Optical flow tracking ---
#         if p0 is not None:
#             p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, gray, p0, None, **lk_params)
#             if st is not None and st[0][0] == 1:
#                 new_pt = tuple(map(int, p1[0][0]))
#                 pts.appendleft(new_pt)
#                 cv2.circle(frame, new_pt, 8, (0, 0, 255), -1)  # tracked fingertip
#                 p0 = np.array([[new_pt]], dtype=np.float32)
#             else:
#                 print("Lost tracking - click inside box to reinitialize")
#                 p0 = None
#                 init_point = None

#             old_gray = gray.copy()

#         # --- Draw smoothed trace ---
#         smoothed = smooth_points(pts)
#         if smoothed and len(pts) >= 2:
#             draw_on_canvas(canvas, pts[1], smoothed)
#             cv2.circle(frame, smoothed, 5, (0, 255, 255), -1)

#         # --- Overlay and FPS ---
#         overlay = overlay_canvas(frame, canvas)
#         curr_time = time.time()
#         fps = 1.0 / (curr_time - prev_time)
#         prev_time = curr_time
#         cv2.putText(overlay, f"FPS: {fps:.1f}", (10, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

#         cv2.imshow('draw', overlay)

#         # --- Controls ---
#         key = cv2.waitKey(1) & 0xFF
#         if key == ord('q'):
#             break
#         elif key == ord('c'):
#             canvas[:] = 0
#             pts.clear()
#             p0 = None
#             init_point = None

#     cap.release()
#     cv2.destroyAllWindows()

# # -------------------- ENTRY POINT --------------------
# if __name__ == "__main__":
#     main()
