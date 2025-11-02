import cv2
import numpy as np
import math
import time
import os
import joblib

class ARDrawingBoard:
    def __init__(self, num_boards=3, camera_index=0):
        # --- Constants ---
        self.NUM_BOARDS = num_boards
        self.DRAW_COLOR = (0, 0, 255)
        self.ERASE_COLOR = (0, 0, 0)
        self.DRAW_RADIUS = 5
        self.ERASE_RADIUS = 20
        self.ALPHA = 0.6
        self.MOVE_ALPHA = 0.5
        self.PREVIEW_SIZE = 100

        # --- Cooldowns ---
        self.PASTE_COOLDOWN = 2.0
        self.last_paste_time = 0.0

        # --- Gesture State (NEW) ---
        self.last_mode_gesture = None
        self.MODE_GESTURE_COOLDOWN = 1.0
        self.last_mode_change_time = 0.0

        # --- NEW: Cooldown for Tag Interaction ---
        self.TAG_COOLDOWN = 1.0
        self.last_tag_trigger_time = 0.0
        
        # --- Shape Check Idle Timer ---
        self.SHAPE_IDLE_TIME = 1.5  # idle time
        self.last_draw_activity_time = 0.0 # Time of last 'draw' event
        self.shape_check_needed = False # Flag to run check after idling

        # --- Shape Detection Setup ---
        self.shape_images = {}
        SHAPE_NAMES = ["heart", "square", "triangle", "house"]
        script_dir = os.path.dirname(os.path.abspath(__file__))
        for shape_name in SHAPE_NAMES:
            try:
                img_path = os.path.join(script_dir, f"{shape_name}.png") 
                img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                if img is None:
                    raise FileNotFoundError(f"File not found: {img_path}")
                self.shape_images[shape_name] = img
                print(f"Loaded replacement image: {shape_name}")
            except Exception as e:
                print(f"ERROR: Could not load replacement image '{shape_name}'. {e}")

        # --- Load SVM Model and Scaler ---
        try:
            self.svm_model = joblib.load('svm_model.pkl')
            self.scaler = joblib.load('scaler.pkl')
            self.class_names = joblib.load('class_names.pkl')
            print("Successfully loaded SVM model, scaler, and class names.")
        except FileNotFoundError:
            print("ERROR: Model files not found. Please run train_model.py first.")
            self.svm_model = None 

        # --- State ---
        self.boards = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(self.NUM_BOARDS)]
        self.current_board_index = 0
        self.mode = 'draw'
        self.drawing = False
        self.last_point = None
        self.drawing_source = None  
        self.cursor_pos = None
        self.action_mode = None # Tracks 'draw' vs 'erase'

        # --- Copy Mode ---
        self.copy_mode = False
        self.selection_points = []
        self.clipboard_img = None

        # --- Manipulation Mode ---
        self.manip_mode = False
        self.manip_selection = []
        self.manip_mask = None
        self.manip_img = None
        self.manip_pos = None
        self.moving = False
        self.move_offset = (0, 0)

        # --- Camera ---
        self.cap = cv2.VideoCapture(camera_index)

        # --- Input System ---
        cv2.namedWindow("AR Drawing Board")
        cv2.setMouseCallback("AR Drawing Board", self._mouse_callback)


    # ============================================================
    #  NEW: TAG INTERACTION HANDLER (Unchanged)
    # ============================================================
    def check_tag_interaction(self, detected_tags):
        if self.cursor_pos is None:
            return

        current_time = time.time()
        if (current_time - self.last_tag_trigger_time) < self.TAG_COOLDOWN:
            return
            
        cx, cy = self.cursor_pos

        for marker_id, (x, y, w, h) in detected_tags.items():
            
            if (cx > x) and (cx < x + w) and (cy > y) and (cy < y + h):
                
                if marker_id == 6:
                    print(f"ACTION: Tag {marker_id} hit -> Page Previous")
                    self.trigger_page_prev()
                    self.last_tag_trigger_time = current_time 
                    break 

                elif marker_id == 8:
                    print(f"ACTION: Tag {marker_id} hit -> Page Next")
                    self.trigger_page_next()
                    self.last_tag_trigger_time = current_time 
                    break 
                
                # --- Add more 'elif marker_id == X:' here for other actions ---

    # ============================================================
    #  UNIFIED INPUT HANDLER (Unchanged)
    # ============================================================
    def handle_input_event(self, event):
        x, y = event["x"], event["y"]
        event_type = event["type"]
        board = self.boards[self.current_board_index]
        self.cursor_pos = (x, y) 

        if self.manip_mode and self.manip_img is not None and event_type == "down":
            self._stamp_manip_img()
            return

        if self.manip_mode and self.manip_img is not None:
            self._handle_manip_move_from_event(event)
            return

        if self.copy_mode:
            if event["type"] == "right_down" and self.clipboard_img is not None:
                self._handle_paste_clipboard_from_event(event, board)
            else:
                self._handle_copy_selection_from_event(event)
            return 

        if self.manip_mode and self.manip_img is None:
            self._handle_manip_selection_from_event(event)
            return

        self._handle_draw_erase_from_event(event, board)


    # ============================================================
    #  MOUSE WRAPPER (Unchanged)
    # ============================================================
    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.handle_input_event({"type": "down", "x": x, "y": y, "source": "mouse"})
        elif event == cv2.EVENT_MOUSEMOVE:
            self.handle_input_event({"type": "move", "x": x, "y": y, "source": "mouse"})
        elif event == cv2.EVENT_LBUTTONUP:
            self.handle_input_event({"type": "up", "x": x, "y": y, "source": "mouse"})
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.handle_input_event({"type": "right_down", "x": x, "y": y, "source": "mouse"})
        elif event == cv2.EVENT_RBUTTONUP:
            self.handle_input_event({"type": "right_up", "x": x, "y": y, "source": "mouse"})

    # ============================================================
    #  DRAW & ERASE (Unchanged)
    # ============================================================
    def _handle_draw_erase_from_event(self, event, board):
        x, y = event["x"], event["y"]
        t = event["type"]
        src = event["source"] 

        current_time = time.time() # Get current time

        if t == "down":
            self.drawing = True
            self.drawing_source = src 
            self.last_point = (x, y)
            self.action_mode = 'draw'
            
            self.last_draw_activity_time = current_time
            self.shape_check_needed = True # A drawing has started
        
        elif t == "right_down":
            self.drawing = True
            self.drawing_source = src 
            self.last_point = (x, y)
            self.action_mode = 'erase'

        elif t == "move" and self.drawing:
            if src == self.drawing_source: 
                if self.action_mode == 'draw':
                    cv2.line(board, self.last_point, (x, y), self.DRAW_COLOR, self.DRAW_RADIUS)
                    self.last_draw_activity_time = current_time
                elif self.action_mode == 'erase':
                    cv2.line(board, self.last_point, (x, y), self.ERASE_COLOR, self.ERASE_RADIUS)
                self.last_point = (x, y)
        
        elif t == "up" or t == "right_up":
            if src == self.drawing_source:
                self.drawing = False
                self.drawing_source = None 
                self.last_point = None
                self.action_mode = None
                # Shape check logic is handled in render_overlay

    # ============================================================
    #  COPY SELECTION (Unchanged)
    # ============================================================
    def _handle_copy_selection_from_event(self, event):
        x, y = event["x"], event["y"]
        t = event["type"]
        src = event["source"] 

        if t == "down":
            self.drawing = True
            self.drawing_source = src
            self.selection_points = [(x, y)]
            self.last_point = (x, y)
        elif t == "move" and self.drawing:
            if src == self.drawing_source: 
                self.selection_points.append((x, y))
                self.last_point = (x, y)
        elif t == "up":
            if src == self.drawing_source: 
                self.drawing = False
                self.drawing_source = None
                self.last_point = None

    def finalize_and_copy_selection(self):
        if len(self.selection_points) < 3:
            print("Selection too small")
            self.selection_points.clear()
            return
        board = self.boards[self.current_board_index]
        mask = np.zeros(board.shape[:2], np.uint8)
        pts = np.array(self.selection_points, np.int32)
        cv2.fillPoly(mask, [pts], 255)

        gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
        drawing_mask = (gray != 0).astype(np.uint8) * 255
        selection_mask = cv2.bitwise_and(mask, drawing_mask)

        if cv2.countNonZero(selection_mask) == 0:
            print("No drawing pixels selected")
            self.selection_points.clear()
            return

        x, y, w, h = cv2.boundingRect(selection_mask)
        crop_color = board[y:y+h, x:x+w].copy()
        crop_mask = selection_mask[y:y+h, x:x+w].copy()

        b, g, r = cv2.split(crop_color)
        self.clipboard_img = cv2.merge((b, g, r, crop_mask))
        print(f"Copied {w}x{h}")
        self.selection_points.clear()

    def _handle_paste_clipboard_from_event(self, event, board):
        if event["type"] == "right_down":
            if event["source"] != "mouse":
                current_time = time.time()
                if (current_time - self.last_paste_time) < self.PASTE_COOLDOWN:
                    print(f"Paste skipped (cooldown).")
                    return 
                self.last_paste_time = current_time
            self._paste_clipboard(board, self.clipboard_img, event["x"], event["y"])

    def _paste_clipboard(self, board, img, x, y):
        h, w = img.shape[:2]
        x1 = max(0, x - w // 2)
        y1 = max(0, y - h // 2)
        x2 = min(board.shape[1], x1 + w)
        y2 = min(board.shape[0], y1 + h)
        src_x2 = x2 - x1
        src_y2 = y2 - y1
        roi = board[y1:y2, x1:x2]
        src = img[0:src_y2, 0:src_x2, :3].astype(np.float32)
        alpha = img[0:src_y2, 0:src_x2, 3:4].astype(np.float32) / 255.0
        roi[:] = (src * alpha + roi.astype(np.float32) * (1 - alpha)).astype(np.uint8)

    # ============================================================
    #  MANIPULATION (Unchanged)
    # ============================================================
    def _handle_manip_selection_from_event(self, event):
        x, y = event["x"], event["y"]
        t = event["type"]
        src = event["source"] 

        if t == "down":
            self.drawing = True
            self.drawing_source = src
            self.manip_selection = [(x, y)]
            self.last_point = (x, y)
        elif t == "move" and self.drawing:
            if src == self.drawing_source: 
                self.manip_selection.append((x, y))
                self.last_point = (x, y)
        elif t == "up":
            if src == self.drawing_source: 
                self.drawing = False
                self.drawing_source = None
                self.last_point = None

    def finalize_manip_selection(self):
        if len(self.manip_selection) < 3:
            print("Manipulation selection too small")
            self.manip_selection.clear()
            return
        board = self.boards[self.current_board_index]
        mask = np.zeros(board.shape[:2], np.uint8)
        pts = np.array(self.manip_selection, np.int32)
        cv2.fillPoly(mask, [pts], 255)
        gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
        drawing_mask = (gray != 0).astype(np.uint8) * 255
        selection_mask = cv2.bitwise_and(mask, drawing_mask)
        if cv2.countNonZero(selection_mask) == 0:
            print("No pixels selected")
            self.manip_selection.clear()
            return
        x, y, w, h = cv2.boundingRect(selection_mask)
        self.manip_img = board[y:y+h, x:x+w].copy()
        self.manip_mask = selection_mask[y:y+h, x:x+w].copy()
        board[y:y+h, x:x+w][self.manip_mask > 0] = 0
        self.manip_pos = (x + w // 2, y + h // 2)
        print(f"Selected area {w}x{h} for manipulation")
        self.manip_selection.clear()

    def _handle_manip_move_from_event(self, event):
        x, y = event["x"], event["y"]
        t = event["type"]
        if t == "right_down":
            self.moving = True
            self.move_offset = (x, y)
        elif t == "move" and self.moving:
            dx = x - self.move_offset[0]
            dy = y - self.move_offset[1]
            self.move_offset = (x, y)
            self.manip_pos = (self.manip_pos[0] + dx, self.manip_pos[1] + dy)
        elif t == "right_up":
            self.moving = False

    def _stamp_manip_img(self):
        if self.manip_img is not None:
            board = self.boards[self.current_board_index]
            h, w = self.manip_img.shape[:2]
            x_off, y_off = self.manip_pos
            x1 = max(0, x_off - w // 2)
            y1 = max(0, y_off - h // 2)
            x2 = min(board.shape[1], x1 + w)
            y2 = min(board.shape[0], y1 + h)
            src_x2 = x2 - x1
            src_y2 = y2 - y1
            roi = board[y1:y2, x1:x2]
            img_section = self.manip_img[0:src_y2, 0:src_x2]
            mask_section = self.manip_mask[0:src_y2, 0:src_x2] / 255.0
            roi[:] = (img_section * mask_section[..., None] + roi * (1 - mask_section[..., None])).astype(np.uint8)
            self.manip_img = None
            self.manip_mask = None
            self.manip_pos = None
            self.manip_selection.clear()

    # ============================================================
    #  MAIN LOOP (Unchanged)
    # ============================================================
    def render_overlay(self, frame):
        
        # --- Idle Shape Check ---
        current_time = time.time()
        if self.shape_check_needed and \
           (current_time - self.last_draw_activity_time) > self.SHAPE_IDLE_TIME:
            
            if not self.copy_mode and not self.manip_mode:
                # This print statement will let you know the check is starting
                print(f"Idle for {self.SHAPE_IDLE_TIME}s, checking for shapes...")
                board = self.boards[self.current_board_index]
                self._check_for_shapes(board) 
            
            self.shape_check_needed = False 

        # --- Original render_overlay code ---
        overlay = self.boards[self.current_board_index].copy()
        self._draw_copy_highlight(overlay)
        self._draw_manip_highlight(overlay)
        combined = cv2.addWeighted(frame, 1 - self.ALPHA, overlay, self.ALPHA, 0)
        self._draw_preview(combined)
        self._draw_mode_text(combined)

        if self.cursor_pos:
            cursor_color = (255, 255, 255)  
            if self.drawing and self.action_mode == 'draw':
                cursor_color = self.DRAW_COLOR
            cv2.circle(combined, self.cursor_pos, 5, cursor_color, -1)
            cv2.circle(combined, self.cursor_pos, 5, (0,0,0), 1)
        return combined

    # ============================================================
    #  UI DRAWING HELPERS (Unchanged)
    # ============================================================
    def _draw_mode_text(self, img):
        mode_str = self.mode.upper()
        text = f"Mode: {mode_str} | Board: {self.current_board_index + 1}/{self.NUM_BOARDS}"
        cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def _draw_copy_highlight(self, overlay):
        if self.copy_mode and self.selection_points:
            pts = np.array(self.selection_points, np.int32)
            if len(pts) >= 2:
                cv2.polylines(overlay, [pts], False, (0, 255, 255), 2)
            if len(pts) >= 3:
                mask = np.zeros(overlay.shape[:2], np.uint8)
                cv2.fillPoly(mask, [pts], 255)
                gray = cv2.cvtColor(self.boards[self.current_board_index], cv2.COLOR_BGR2GRAY)
                drawing_mask = (gray != 0).astype(np.uint8) * 255
                highlight = cv2.bitwise_and(mask, drawing_mask)
                overlay[highlight > 0] = (0, 255, 0)

    def _draw_manip_highlight(self, overlay):
        if self.manip_mode:
            if self.manip_img is not None:
                self._blend_manip_image(overlay)
            elif self.manip_selection:
                pts = np.array(self.manip_selection, np.int32)
                if len(pts) >= 2:
                    cv2.polylines(overlay, [pts], False, (255, 0, 0), 2)
                if len(pts) >= 3:
                    mask = np.zeros(overlay.shape[:2], np.uint8)
                    cv2.fillPoly(mask, [pts], 255)
                    gray = cv2.cvtColor(self.boards[self.current_board_index], cv2.COLOR_BGR2GRAY)
                    drawing_mask = (gray != 0).astype(np.uint8) * 255
                    highlight = cv2.bitwise_and(mask, drawing_mask)
                    overlay[highlight > 0] = (255, 0, 0)

    def _blend_manip_image(self, overlay):
        temp = overlay.copy()
        h, w = self.manip_img.shape[:2]
        x_off, y_off = self.manip_pos
        x1 = max(0, x_off - w // 2)
        y1 = max(0, y_off - h // 2)
        x2 = min(temp.shape[1], x1 + w)
        y2 = min(temp.shape[0], y1 + h)
        src_x2 = x2 - x1
        src_y2 = y2 - y1
        roi = temp[y1:y2, x1:x2]
        img_section = self.manip_img[0:src_y2, 0:src_x2].astype(np.float32)
        mask_section = self.manip_mask[0:src_y2, 0:src_x2].astype(np.float32) / 255.0 * self.MOVE_ALPHA
        roi[:] = (img_section * mask_section[..., None] + roi.astype(np.float32) * (1 - mask_section[..., None])).astype(np.uint8)
        overlay[:] = temp

    def _draw_preview(self, combined):
        preview_source = None
        if self.manip_mode and self.manip_img is not None:
            b, g, r = cv2.split(self.manip_img)
            preview_source = cv2.merge((b, g, r, self.manip_mask))
        elif self.clipboard_img is not None:
            preview_source = self.clipboard_img
        if preview_source is not None:
            clip_h, clip_w = preview_source.shape[:2]
            scale = self.PREVIEW_SIZE / max(clip_h, clip_w)
            new_w, new_h = int(clip_w * scale), int(clip_h * scale)
            preview = cv2.resize(preview_source, (new_w, new_h), interpolation=cv2.INTER_AREA)
            preview_bgr = preview[:, :, :3]
            preview_alpha = preview[:, :, 3] / 255.0
            x_offset = combined.shape[1] - new_w - 10
            y_offset = 10
            roi = combined[y_offset:y_offset + new_h, x_offset:x_offset + new_w]
            alpha_mask = preview_alpha[..., None]
            roi[:] = (preview_bgr * alpha_mask + roi * (1 - alpha_mask)).astype(np.uint8)
            cv2.rectangle(combined, (x_offset, y_offset), (x_offset + new_w, y_offset + new_h), (255, 255, 255), 1)

    # ============================================================
    #  KEY INPUT HANDLER (Unchanged)
    # ============================================================
    
    def _handle_key_input(self, key):
        if key == 27:  # ESC
            return False

        if key == ord('q'):
            self.trigger_page_prev() 
            return True 
        elif key == ord('w'):
            self.trigger_page_next() 
            return True 
            
        if key == ord('d'):
            self.set_mode_draw()

        elif key == ord('c'):
            self.set_mode_copy()

        elif key == ord('m'):
            self.set_mode_manip()

        elif key == ord('s'):
            self.trigger_selection_finalize() 

        elif key == 32:  # SPACE to clear current board
            self.boards[self.current_board_index][:] = 0
            self.shape_check_needed = False 

        return True

    # ============================================================
    #  SHAPE DETECTION (MODIFIED)
    # ============================================================
    @staticmethod
    def _overlay_transparent_image(background, overlay, x, y):
        if overlay.shape[2] < 4: return
        h, w, _ = background.shape
        h_overlay, w_overlay, _ = overlay.shape
        x1, x2 = max(0, x), min(w, x + w_overlay)
        y1, y2 = max(0, y), min(h, y + h_overlay)
        x1_overlay, x2_overlay = max(0, -x), min(w_overlay, w - x)
        y1_overlay, y2_overlay = max(0, -y), min(h_overlay, h - y)
        if x1 >= x2 or y1 >= y2: return
        alpha = overlay[y1_overlay:y2_overlay, x1_overlay:x2_overlay, 3] / 255.0
        alpha_inv = 1.0 - alpha
        overlay_rgb = overlay[y1_overlay:y2_overlay, x1_overlay:x2_overlay, :3]
        for c in range(0, 3):
            background[y1:y2, x1:x2, c] = (alpha * overlay_rgb[:, :, c] +
                                          alpha_inv * background[y1:y2, x1:x2, c])

    def _get_features_from_contour(self, contour):
        moments = cv2.moments(contour)
        hu_moments = cv2.HuMoments(moments)
        for i in range(7):
            hu_moments[i] = -1 * math.copysign(1.0, hu_moments[i][0]) * math.log10(abs(hu_moments[i][0]) + 1e-9)
        return hu_moments.flatten()
    
    def _check_for_shapes(self, board):
        """
        Finds contours for all separate blobs, predicts them, and replaces them.
        Uses MORPH_OPEN to separate blobs that are drawn too close.
        """
        if self.svm_model is None or not self.shape_images:
            print("  -> Shape check skipped: Model not loaded.")
            return

        print("\n--- Checking for shapes (Blob Method) ---")
        canvas_gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(canvas_gray, 10, 255, cv2.THRESH_BINARY)
        
        # --- FIXED: Use a less aggressive kernel ---
        # A 3x3 kernel with 1 iteration is safer.
        # It erodes 1 pixel, then dilates 1 pixel, just enough
        # to break thin connections without destroying shapes.
        kernel = np.ones((3, 3), np.uint8) 
        thresh_opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        
        contours, _ = cv2.findContours(thresh_opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        replacements = []
        contours_found_and_kept = 0 # Debug counter
        
        for contour in contours:
            
            if cv2.contourArea(contour) < 1500: # Tune this area value if needed
                continue
            
            contours_found_and_kept += 1 # Add to debug counter
            features = self._get_features_from_contour(contour)
            features_reshaped = features.reshape(1, -1)
            scaled_features = self.scaler.transform(features_reshaped)
            prediction_proba = self.svm_model.predict_proba(scaled_features)[0]
            confidence = prediction_proba.max()
            prediction_index = prediction_proba.argmax()
            shape_name = self.class_names[prediction_index] 
            
            CONF_THRESHOLD = 0.80 
            if confidence >= CONF_THRESHOLD:
                print(f"  -> Matched {shape_name.upper()} (Confidence: {confidence*100:.1f}%)")
                if shape_name in self.shape_images:
                    replacements.append((contour, shape_name))
            else:
                 print(f"  -> Low confidence match: {shape_name.upper()} (Confidence: {confidence*100:.1f}%)")

        # --- NEW: Debug print ---
        print(f"  -> Found {len(contours)} total blobs, kept {contours_found_and_kept} after filtering.")

        for contour, shape_name in replacements:
            x, y, w_drawn, h_drawn = cv2.boundingRect(contour)
            
            # Erase the blob from the *original* board
            # We must use the contour found on the 'opened' image
            cv2.drawContours(board, [contour], -1, self.ERASE_COLOR, cv2.FILLED)
            
            replacement_img = self.shape_images[shape_name]
            
            padding_factor = 1.2 
            new_w = int(w_drawn * padding_factor)
            new_h = int(h_drawn * padding_factor)
            
            if new_w > 0 and new_h > 0:
                resized_img = cv2.resize(replacement_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                overlay_x = x - (new_w - w_drawn) // 2
                overlay_y = y - (new_h - h_drawn) // 2
                self._overlay_transparent_image(board, resized_img, overlay_x, overlay_y)

    # ============================================================
    #  MODE MANAGEMENT (Unchanged)
    # ============================================================

    def handle_mode_gesture(self, gesture_input):
        current_time = time.time()
        
        if (current_time - self.last_mode_change_time) < self.MODE_GESTURE_COOLDOWN:
            return
        
        base_gesture = gesture_input.get('base_gesture')
        direction = gesture_input.get('direction')
        
        target_mode = None
        
        if base_gesture == 'pinky_up':
            if direction == 'centered':
                target_mode = 'draw'
            elif direction == 'left':
                target_mode = 'copy'
            elif direction == 'right':
                target_mode = 'manip' 
        
        if target_mode and target_mode != self.mode:
            
            if target_mode != self.last_mode_gesture: 
                
                if target_mode == 'draw':
                    self.set_mode_draw()
                elif target_mode == 'copy':
                    self.set_mode_copy()
                elif target_mode == 'manip':
                    self.set_mode_manip()
                    
                self.last_mode_gesture = target_mode
                self.last_mode_change_time = current_time
        
        if base_gesture != 'pinky_up':
            self.last_mode_gesture = None

    def _reset_modes(self, draw=False):
        self.copy_mode = False
        self.manip_mode = False
        self.selection_points.clear()
        self.manip_selection.clear()
        self.clipboard_img = None
        self.manip_img = None
        self.manip_mask = None
        self.moving = False
        self.move_offset = (0, 0)
        self.drawing = False
        self.drawing_source = None 
        self.last_point = None
        self.manip_pos = None
        self.action_mode = None
        
        self.shape_check_needed = False

        if draw:
            self.mode = 'draw'
        else:
            self.mode = 'draw'

    # --- NEW: Public methods for mode switching ---

    def set_mode_draw(self):
        """Sets the board to the default 'draw' mode."""
        self._stamp_manip_img()
        self._reset_modes(draw=True)
        print("Mode set to DRAW")

    def set_mode_copy(self):
        """Toggles the 'copy' mode."""
        self._stamp_manip_img()
        if self.copy_mode:
            self._reset_modes(draw=True) 
            print("Mode set to DRAW (toggled off copy)")
        else:
            self._reset_modes()
            self.copy_mode = True
            self.mode = 'copy'
            print("Mode set to COPY")

    def set_mode_manip(self):
        """Toggles the 'manipulate' mode."""
        self._stamp_manip_img()
        if self.manip_mode:
            self._reset_modes(draw=True) 
            print("Mode set to DRAW (toggled off manipulate)")
        else:
            self._reset_modes()
            self.manip_mode = True
            self.mode = 'manipulate'
            print("Mode set to MANIPULATE")

    # --- NEW: Public methods for actions (MODIFIED) ---

    def trigger_page_next(self):
        """Switches to the next board."""
        self.current_board_index = (self.current_board_index + 1) % self.NUM_BOARDS
        print(f"Switched to board {self.current_board_index + 1}")

    def trigger_page_prev(self):
        """Switches to the previous board."""
        self.current_board_index = (self.current_board_index - 1) % self.NUM_BOARDS
        print(f"Switched to board {self.current_board_index + 1}")

    def trigger_selection_finalize(self):
        """
        Public method to finalize a selection.
        Called by 'S' key or Fist gesture.
        """
        if self.copy_mode:
            print("Action: Finalizing copy selection")
            self.finalize_and_copy_selection()
        elif self.manip_mode and self.manip_img is None:
            print("Action: Finalizing manip selection")
            self.finalize_manip_selection()