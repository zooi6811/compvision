import cv2
import mediapipe as mp
import numpy as np
import math
import time

# --- Constants ---
DRAW_COLOUR = (255, 0, 0)
ERASE_COLOUR = (0, 0, 0)
CURSOR_COLOUR = (0, 255, 0)
RESIZE_LINE_COLOUR = (0, 255, 255)

# --- Brush Size ---
MIN_THICKNESS = 5
MAX_THICKNESS = 100
ERASER_MULTIPLIER = 3
PINCH_DIST_MIN = 30
PINCH_DIST_MAX = 200
active_thickness = 15.0  # MODIFIED: Changed to float for smoothing
SMOOTHING_FACTOR = 0.8 # NEW: 80% old value, 20% new value. HIGHER = smoother.

# --- MediaPipe Setup ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
tip_ids = [4, 8, 12, 16, 20]

# --- Heart PNG ---
try:
    heart_png = cv2.imread('heart.png', cv2.IMREAD_UNCHANGED)
    if heart_png is None:
        raise FileNotFoundError("heart.png not found or could not be loaded.")
    print("Heart PNG loaded successfully.")
except FileNotFoundError as e:
    print(f"Error: {e}. Defaulting to drawing a simple red circle if heart.png is not found.")
    heart_png = None

is_drawing_active = False

# --- Mode State Variables ---
meta_mode = "Select"
mode_change_time = 0.0

# --- (check_fingers_up function is unchanged) ---
def check_fingers_up(hand_landmarks, handedness):
    fingers = []
    landmarks = hand_landmarks.landmark
    is_right_hand = (handedness.classification[0].label == 'Right')
    if is_right_hand:
        if landmarks[tip_ids[0]].x < landmarks[tip_ids[0] - 2].x:
            fingers.append(True)
        else:
            fingers.append(False)
    else: # Left Hand
        if landmarks[tip_ids[0]].x > landmarks[tip_ids[0] - 2].x:
            fingers.append(True)
        else:
            fingers.append(False)
    for finger_id in range(1, 5):
        if landmarks[tip_ids[finger_id]].y < landmarks[tip_ids[finger_id] - 2].y:
            fingers.append(True)
        else:
            fingers.append(False)
    return fingers

# --- (overlay_transparent_image function is unchanged) ---
def overlay_transparent_image(background, overlay, x, y):
    if overlay.shape[2] < 4: return background
    h, w, _ = background.shape
    h_overlay, w_overlay, _ = overlay.shape
    x1, x2 = max(0, x), min(w, x + w_overlay); y1, y2 = max(0, y), min(h, y + h_overlay)
    x1_overlay, x2_overlay = max(0, -x), min(w_overlay, w - x)
    y1_overlay, y2_overlay = max(0, -y), min(h_overlay, h - y)
    if x1 >= x2 or y1 >= y2: return background
    alpha = overlay[y1_overlay:y2_overlay, x1_overlay:x2_overlay, 3] / 255.0
    alpha_inv = 1.0 - alpha; overlay_rgb = overlay[y1_overlay:y2_overlay, x1_overlay:x2_overlay, :3]
    for c in range(0, 3):
        background[y1:y2, x1:x2, c] = (alpha * overlay_rgb[:, :, c] + alpha_inv * background[y1:y2, x1:x2, c])
    return background

# --- (is_heart_shape function is unchanged) ---
def is_heart_shape(contour):
    if len(contour) < 50: return False
    area = cv2.contourArea(contour)
    if area < 1500: return False
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0: return False
    epsilon = 0.02 * perimeter; approx = cv2.approxPolyDP(contour, epsilon, True)
    num_vertices = len(approx); x, y, w, h = cv2.boundingRect(contour); aspect_ratio = float(w) / h
    if not (0.7 < aspect_ratio < 1.3): return False
    try:
        hull = cv2.convexHull(contour, returnPoints=False)
        if len(hull) < 3: return False
        defects = cv2.convexityDefects(contour, hull)
        if defects is None: return False
    except cv2.error: return False
    if 9 <= num_vertices <= 15: return True
    return False

# --- Main Program ---
def main():
    global is_drawing_active, active_thickness
    global meta_mode, mode_change_time

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    with mp_hands.Hands(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
        max_num_hands=1) as hands:

        canvas = None
        prev_draw_action = "Select"
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame."); continue

            image = cv2.flip(image, 1)
            
            if canvas is None:
                # Get the height (h) and width (w)
                h, w, _ = image.shape
                canvas = np.zeros((h, w, 3), dtype=np.uint8)

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            current_drawing_state = False
            ui_text = f"Mode: {meta_mode}"

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                handedness = results.multi_handedness[0]
                fingers_up = check_fingers_up(hand_landmarks, handedness)

                # We need 'h' and 'w' here from the image shape
                h, w, _ = image.shape
                index_tip = hand_landmarks.landmark[tip_ids[1]]
                cx, cy = int(index_tip.x * w), int(index_tip.y * h)

                gest_index_pinky = [False, True, False, False, True]
                gest_index = [False, True, False, False, False]
                gest_index_middle = [False, True, True, False, False]
                
                if fingers_up == gest_index_pinky and meta_mode != "Waiting":
                    meta_mode = "Waiting"
                    mode_change_time = time.time()
                    print("ENTERING MODE SELECT: Hold new gesture for 5s...")

                if meta_mode == "Waiting":
                    elapsed = time.time() - mode_change_time
                    countdown = 5.0 - elapsed
                    
                    if countdown <= 0:
                        print("Selecting mode...")
                        if fingers_up == gest_index:
                            meta_mode = "Drawing"
                        elif fingers_up == gest_index_middle:
                            meta_mode = "Resizing"
                        else:
                            meta_mode = "Select"
                        print(f"Switched to {meta_mode}")
                    else:
                        ui_text = f"Waiting... {countdown:.1f}s"
                        cv2.circle(image, (cx, cy), 10, (0, 0, 255), 2)
                
                elif meta_mode == "Drawing":
                    current_action = "Select"
                    
                    if fingers_up == gest_index:
                        cv2.circle(canvas, (cx, cy), int(active_thickness), DRAW_COLOUR, -1)
                        current_drawing_state = True
                        current_action = "Draw"
                        
                    elif fingers_up == gest_index_middle:
                        eraser_size = int(int(active_thickness) * ERASER_MULTIPLIER)
                        cv2.circle(canvas, (cx, cy), eraser_size, ERASE_COLOUR, -1)
                        current_action = "Erase"
                        
                    else:
                        cv2.circle(image, (cx, cy), int(active_thickness), CURSOR_COLOUR, 2)
                        
                    if prev_draw_action == "Draw" and not current_drawing_state and is_drawing_active:
                        print("Finished drawing. Checking for shapes...")
                        canvas_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
                        _, thresh = cv2.threshold(canvas_gray, 10, 255, cv2.THRESH_BINARY)
                        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        for contour in contours:
                            if is_heart_shape(contour) and heart_png is not None:
                                print("Heart shape detected!")
                                x, y, w_drawn, h_drawn = cv2.boundingRect(contour)
                                cv2.drawContours(canvas, [contour], -1, ERASE_COLOUR, cv2.FILLED)
                                padding_factor = 1.2; new_w = int(w_drawn * padding_factor); new_h = int(h_drawn * padding_factor)
                                if new_w > 0 and new_h > 0:
                                    resized_heart = cv2.resize(heart_png, (new_w, new_h), interpolation=cv2.INTER_AREA)
                                    overlay_x = x - (new_w - w_drawn) // 2; overlay_y = y - (new_h - h_drawn) // 2
                                    canvas = overlay_transparent_image(canvas, resized_heart, overlay_x, overlay_y)
                    
                    is_drawing_active = current_drawing_state
                    prev_draw_action = current_action

                # --- MODE: Resizing (WITH THE FIX) ---
                elif meta_mode == "Resizing":
                    
                    if fingers_up == gest_index_middle: 
                        
                        # 1. Calculate the target thickness based on Y-position
                        #    MODIFIED: Map the *entire* screen height
                        #    [0, h] -> [MAX, MIN]
                        #    This means 0 (top) is MAX size, h (bottom) is MIN size.
                        target_thickness = np.interp(cy, [0, h], [MAX_THICKNESS, MIN_THICKNESS])
                        
                        # 2. Apply smoothing
                        active_thickness = (active_thickness * SMOOTHING_FACTOR) + (target_thickness * (1.0 - SMOOTHING_FACTOR))
                        
                        # 3. Clip just in case
                        active_thickness = np.clip(active_thickness, MIN_THICKNESS, MAX_THICKNESS)
                        
                        # 4. Show preview circle
                        middle_tip = hand_landmarks.landmark[tip_ids[2]]
                        mx, my = int(middle_tip.x * w), int(middle_tip.y * h)
                        cv2.circle(image, (mx, my), int(active_thickness), CURSOR_COLOUR, 2)

                    else:
                        cv2.circle(image, (cx, cy), int(active_thickness), CURSOR_COLOUR, 2)

                # --- MODE: Select ---
                elif meta_mode == "Select":
                    cv2.circle(image, (cx, cy), int(active_thickness), CURSOR_COLOUR, 2)

            # --- Combine Canvas and Webcam Image ---
            img_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
            _, img_inv = cv2.threshold(img_gray, 10, 255, cv2.THRESH_BINARY_INV)
            img_bg = cv2.bitwise_and(image, image, mask=img_inv)
            final_image = cv2.add(img_bg, canvas)
            
            # --- Display UI ---
            cv2.putText(final_image, ui_text, (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(final_image, f"Size: {int(active_thickness)}", (20, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow('AR Whiteboard', final_image)

            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()