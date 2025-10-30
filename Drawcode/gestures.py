import cv2
import numpy as np
from collections import deque, Counter
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

FINGER_JOINT_SETS = {
    "Thumb": [1, 2, 3, 4],
    "Index": [5, 6, 7, 8],
    "Middle": [9, 10, 11, 12],
    "Ring": [13, 14, 15, 16],
    "Pinky": [17, 18, 19, 20],
}


class FaceHandTracker:
    def __init__(self, board=None): # <-- MODIFIED: Accept board object

        self.board = board # <-- NEW: Store board object
        # <-- NEW: Get event callback from the board
        self.event_callback = board.handle_input_event if board else None 

        self.last_gesture = "None"
        self.previous_gesture = "None"
        self.previous_pos = None
        self.gesture_buffer = deque(maxlen=10)
        self.frame_count = 0
        # self.event_callback = event_callback # <-- REMOVED

        self.face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        self.hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)

    def process_frame(self, image):
        ih, iw, _ = image.shape
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False

        self.face_detection.process(image_rgb)
        hand_results = self.hands.process(image_rgb)
        image_rgb.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                self.process_hand(image, hand_landmarks.landmark, iw, ih)

        self.frame_count += 1
        if self.frame_count >= 30:
            majority_gesture = self.get_majority_gesture()
            # print(f"Majority Gesture: {majority_gesture}") # Optional: uncomment for debugging
            self.frame_count = 0
            self.gesture_buffer.clear()

        return image

    def process_hand(self, image, landmarks, iw, ih):
        # bounding box just for visualization
        x_vals = [int(lm.x * iw) for lm in landmarks]
        y_vals = [int(lm.y * ih) for lm in landmarks]
        x_min, x_max = max(min(x_vals) - 20, 0), min(max(x_vals) + 20, iw)
        y_min, y_max = max(min(y_vals) - 20, 0), min(max(y_vals) + 20, ih)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)

        # analyze hand
        direction, tilt, distance = self.estimate_hand_orientation_2d(landmarks, iw, ih)
        thumb_dir, thumb_tilt = self.estimate_thumb_orientation_2d(landmarks, iw, ih)
        fingers = self.fingers_status(landmarks)
        gesture = self.gesture_recognise(direction, thumb_dir, tilt, thumb_tilt, fingers)
        self.last_gesture = gesture
        self.gesture_buffer.append(gesture)

        # fingertip position (use index finger tip for all actions)
        x = int(landmarks[8].x * iw)
        y = int(landmarks[8].y * ih)

        # smoothing
        if self.previous_pos is not None:
            alpha = 0.4
            x = int(alpha * x + (1 - alpha) * self.previous_pos[0])
            y = int(alpha * y + (1 - alpha) * self.previous_pos[1])

        # --- MODIFIED: Gesture-to-event mapping ---
        
        event = None

        # 1. Set the board's mode based on the gesture
        if self.board and not self.board.copy_mode and not self.board.manip_mode:
            if gesture == "Finger_Point":
                self.board.mode = 'draw'
            elif gesture == "Erase_Gesture":
                self.board.mode = 'erase'
            # Note: If gesture is Open_Palm or Fist, the mode remains what it was.
            # This allows you to lift your pen (fist) and put it back down
            # in the same mode without it defaulting back to 'draw'.

        # 2. Create down/move/up events based on the gesture
        
        # 🖊 Drawing mode: finger point acts as pen tip
        if gesture == "Finger_Point":
            if self.previous_gesture != "Finger_Point":
                # finger started pointing → pen down
                event = {"type": "down", "x": x, "y": y, "source": "gesture"}
            else:
                # continue drawing
                event = {"type": "move", "x": x, "y": y, "source": "gesture"}
        
        # ✌️ Erasing mode: index and middle finger
        elif gesture == "Erase_Gesture":
            if self.previous_gesture != "Erase_Gesture":
                # Erase gesture started → pen down (in erase mode)
                event = {"type": "down", "x": x, "y": y, "source": "gesture"}
            else:
                # continue erasing
                event = {"type": "move", "x": x, "y": y, "source": "gesture"}

        # 🖐 Hover mode: open palm moves cursor but doesn't draw
        elif gesture == "Open_Palm":
            # If we were just drawing OR erasing, send an 'up' event first
            if self.previous_gesture == "Finger_Point" or self.previous_gesture == "Erase_Gesture":
                event = {"type": "up", "x": x, "y": y, "source": "gesture"}
                if self.event_callback:
                    self.event_callback(event) # Send the 'up' event immediately
            
            # Now, always send a 'move' event for the hover
            event = {"type": "move", "x": x, "y": y, "source": "gesture"}

        # ✊ Any other gesture: if we were drawing/erasing, lift pen up
        else:
            if self.previous_gesture == "Finger_Point" or self.previous_gesture == "Erase_Gesture":
                event = {"type": "up", "x": x, "y": y, "source": "gesture"}

        # --- End of modified section ---

        # Send event to drawing system if callback provided
        if event and self.event_callback:
            self.event_callback(event)

        self.previous_gesture = gesture
        self.previous_pos = (x, y)

        # debug info
        # cv2.putText(image, f"Gesture: {gesture}", (10, 30),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 255), 2)
        
        # --- NEW: Show current board mode on screen ---
        # if self.board:
        #      cv2.putText(image, f"Mode: {self.board.mode.upper()}", (10, 60),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 255), 2)


    def get_majority_gesture(self):
        if not self.gesture_buffer:
            return "None"
        return Counter(self.gesture_buffer).most_common(1)[0][0]

    def estimate_hand_orientation_2d(self, lm, iw, ih):
        x5, y5 = lm[5].x * iw, lm[5].y * ih
        x17, y17 = lm[17].x * iw, lm[17].y * ih
        x0, y0 = lm[0].x * iw, lm[0].y * ih
        x9, y9 = lm[9].x * iw, lm[9].y * ih
        x13, y13 = lm[13].x * iw, lm[13].y * ih
        x_avg = (x5 + x17) / 2
        y_avg = (y5 + y17) / 2
        dist = np.hypot(x9 - x13, y9 - y13) / 20
        dx, dy = x0 - x_avg, y0 - y_avg

        direction = 'right' if dx < (-30 * dist) else 'left' if dx > (30 * dist) else 'centered'
        tilt = 'up' if dy > (30 * dist) else 'down' if dy < (-30 * dist) else 'level'
        return direction, tilt, dist

    def estimate_thumb_orientation_2d(self, lm, iw, ih):
        x0, y0 = lm[0].x * iw, lm[0].y * ih
        x2, y2 = lm[2].x * iw, lm[2].y * ih
        x9, y9 = lm[9].x * iw, lm[9].y * ih
        x13, y13 = lm[13].x * iw, lm[13].y * ih
        dist = np.hypot(x9 - x13, y9 - y13) / 20
        dx, dy = x0 - x2, y0 - y2

        direction = 'left' if dx > (30 * dist) else 'right' if dx < (-30 * dist) else 'neutral'
        tilt = 'up' if dy > (30 * dist) else 'down' if dy < (-30 * dist) else 'level'
        return direction, tilt

    def fingers_status(self, lm):
        status = []
        for finger, joints in FINGER_JOINT_SETS.items():
            threshold = 30 if finger == "Thumb" else 20
            pts = [np.array([lm[i].x, lm[i].y, lm[i].z]) for i in joints]
            vecs = [pts[i+1] - pts[i] for i in range(len(pts)-1)]
            vecs = [v / np.linalg.norm(v) for v in vecs]
            straight = all(
                np.degrees(np.arccos(np.clip(np.dot(vecs[i], vecs[i+1]), -1.0, 1.0))) < threshold
                for i in range(len(vecs)-1)
            )
            status.append(1 if straight else 0)
        return status

    def gesture_recognise(self, hand_direction, thumb_direction, hand_tilt, thumb_tilt, fingers):
        gesture = "Unknown"
        # --- MODIFIED: Renamed "Peace" to "Erase_Gesture" ---
        if fingers == [0, 1, 1, 0, 0]:
            gesture = "Erase_Gesture" 
        elif fingers == [1, 1, 1, 1, 1]:
            gesture = "Open_Palm"
        elif fingers == [0, 0, 0, 0, 0]:
            gesture = "Fist"
        elif fingers == [0, 1, 0, 0, 1] and hand_tilt == "up":
            gesture = "Spider"
        elif fingers == [1, 0, 0, 0, 0]:
            if thumb_tilt == "up":
                gesture = "Thumbs_Up"
            elif thumb_tilt == "down":
                gesture = "Thumbs_Down"
            elif thumb_tilt == "right":
                gesture = "Thumbs_Right"
            elif thumb_tilt == "left":
                gesture = "Thumbs_Left"
        elif fingers == [0, 1, 0, 0, 0] or fingers == [1, 1, 0, 0, 0]:
            # regardless of orientation
            gesture = "Finger_Point"
        elif fingers == [0, 0, 1, 0, 0] or fingers == [1, 0, 1, 0, 0]:
            if hand_tilt == "up":
                gesture = "Assert_Dominance"
        return gesture