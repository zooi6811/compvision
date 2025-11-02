import cv2
import numpy as np
from collections import deque, Counter
import mediapipe as mp
import time # Import time for cooldown management

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
    def __init__(self, board=None): 
        self.board = board 
        # Note: self.event_callback is for drawing/erasing events.
        self.event_callback = board.handle_input_event if board else None 

        self.last_gesture = "None"
        self.previous_gesture = "None"
        self.previous_pos = None
        self.gesture_buffer = deque(maxlen=10)
        self.frame_count = 0

        # NEW: Variables to store the latest processed hand data for mode switching
        self.last_mode_data = {'base_gesture': None, 'direction': None} 
        self.last_hand_direction = 'centered' # Default
        self.last_fingers_status = [0, 0, 0, 0, 0] # Default

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

        # Default mode data if no hand is detected
        self.last_mode_data = {'base_gesture': None, 'direction': None} 

        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                self.process_hand(image, hand_landmarks.landmark, iw, ih)

        # --- NEW: Call mode gesture handler on the board ---
        if self.board:
            self.board.handle_mode_gesture(self.last_mode_data)
        # --- END NEW ---

        self.frame_count += 1
        if self.frame_count >= 30:
            majority_gesture = self.get_majority_gesture()
            self.frame_count = 0
            self.gesture_buffer.clear()

        return image

    def process_hand(self, image, landmarks, iw, ih):
        x_vals = [int(lm.x * iw) for lm in landmarks]
        y_vals = [int(lm.y * ih) for lm in landmarks]
        x_min, x_max = max(min(x_vals) - 20, 0), min(max(x_vals) + 20, iw)
        y_min, y_max = max(min(y_vals) - 20, 0), min(max(y_vals) + 20, ih)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)

        direction, tilt, distance = self.estimate_hand_orientation_2d(landmarks, iw, ih)
        thumb_dir, thumb_tilt = self.estimate_thumb_orientation_2d(landmarks, iw, ih)
        fingers = self.fingers_status(landmarks)
        
        # Store orientation and status for mode switching
        self.last_hand_direction = direction
        self.last_fingers_status = fingers
        
        gesture = self.gesture_recognise(direction, thumb_dir, tilt, thumb_tilt, fingers)
        self.last_gesture = gesture
        self.gesture_buffer.append(gesture)

        # --- NEW: Update the mode gesture data dictionary ---
        self.last_mode_data = self.get_mode_gesture_data(gesture, direction)
        # --- END NEW ---

        x = int(landmarks[8].x * iw)
        y = int(landmarks[8].y * ih)

        if self.previous_pos is not None:
            alpha = 0.4
            x = int(alpha * x + (1 - alpha) * self.previous_pos[0])
            y = int(alpha * y + (1 - alpha) * self.previous_pos[1])

        # --- MODIFIED: Check for Fist to finalize selection ---
        # This runs when the gesture *first* becomes a fist
        if gesture == "Fist" and self.previous_gesture != "Fist":
            if self.board:
                # Call the public method on the board
                self.board.trigger_selection_finalize()
                
        # --- Gesture-to-event mapping (Drawing/Erasing) ---
        
        event = None

        # 2. Create down/move/up events based on the gesture
        
        # 🖊 "left-click"
        if gesture == "Finger_Point":
            if self.previous_gesture != "Finger_Point":
                event = {"type": "down", "x": x, "y": y, "source": "gesture"}
            else:
                event = {"type": "move", "x": x, "y": y, "source": "gesture"}
        
        # ✌️ "right-click"
        elif gesture == "Double_Point_Gesture":
            if self.previous_gesture != "Double_Point_Gesture":
                event = {"type": "right_down", "x": x, "y": y, "source": "gesture"}
            else:
                event = {"type": "move", "x": x, "y": y, "source": "gesture"}

        # 🖐 Hover / Release
        elif gesture == "Open_Palm" or gesture == "Pinky_Up": # Added Pinky_Up for smoother release
            if self.previous_gesture == "Finger_Point":
                event = {"type": "up", "x": x, "y": y, "source": "gesture"}
            elif self.previous_gesture == "Double_Point_Gesture":
                event = {"type": "right_up", "x": x, "y": y, "source": "gesture"}
            
            if event and self.event_callback:
                self.event_callback(event) 
            
            # Still send a move event to update the cursor position
            event = {"type": "move", "x": x, "y": y, "source": "gesture"}

        # ✊ Any other gesture (e.g., Fist, Unknown): Lift all pens
        else:
            if self.previous_gesture == "Finger_Point":
                event = {"type": "up", "x": x, "y": y, "source": "gesture"}
            elif self.previous_gesture == "Double_Point_Gesture":
                event = {"type": "right_up", "x": x, "y": y, "source": "gesture"}

        if event and self.event_callback:
            self.event_callback(event)

        self.previous_gesture = gesture
        self.previous_pos = (x, y)

    # ============================================================
    #  NEW MODE GESTURE DATA GENERATOR
    # ============================================================
    
    def get_mode_gesture_data(self, current_gesture, hand_direction):
        """
        Generates the simplified data dictionary required by ARDrawingBoard's mode handler.
        """
        data = {'base_gesture': None, 'direction': None}
        
        if current_gesture == "Pinky_Up":
            data['base_gesture'] = 'pinky_up'
            data['direction'] = hand_direction
            
        return data


    # ============================================================
    #  UPDATED GESTURE RECOGNITION
    # ============================================================

    def gesture_recognise(self, hand_direction, thumb_direction, hand_tilt, thumb_tilt, fingers):
        """
        Detects specific hand gestures based on finger status and orientation.
        """
        
        # fingers list: [Thumb, Index, Middle, Ring, Pinky] (1=straight, 0=bent)

        # --- NEW: Pinky Up for Mode Switching ---
        # Thumb: Bent(0), Index: Bent(0), Middle: Bent(0), Ring: Bent(0), Pinky: Straight(1)
        if fingers[0] == 0 and fingers[1] == 0 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 1:
             return "Pinky_Up"
        # --- END NEW ---

        gesture = "Unknown"
        
        # Double Point / V Sign
        if fingers == [0, 1, 1, 0, 0] or fingers == [1, 1, 1, 0, 0]:
            gesture = "Double_Point_Gesture" 
        # Open Palm
        elif fingers == [1, 1, 1, 1, 1]:
            gesture = "Open_Palm"
        # Fist
        elif fingers == [0, 0, 0, 0, 0]:
            gesture = "Fist"
        # Spider (Index and Pinky up)
        elif fingers == [0, 1, 0, 0, 1] and hand_tilt == "up":
            gesture = "Spider"
        # Thumbs
        elif fingers == [1, 0, 0, 0, 0]:
            if thumb_tilt == "up":
                gesture = "Thumbs_Up"
            elif thumb_tilt == "down":
                gesture = "Thumbs_Down"
            elif thumb_direction == "right":
                gesture = "Thumbs_Right"
            elif thumb_direction == "left":
                gesture = "Thumbs_Left"
        # Single Point
        elif fingers == [0, 1, 0, 0, 0] or fingers == [1, 1, 0, 0, 0]:
            gesture = "Finger_Point"
        # Middle Finger (Assert Dominance)
        elif fingers == [0, 0, 1, 0, 0] or fingers == [1, 0, 1, 0, 0]:
            if hand_tilt == "up":
                gesture = "Assert_Dominance"
                
        return gesture

    # --- (All other helper functions are unchanged) ---
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