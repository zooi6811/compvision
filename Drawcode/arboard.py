import cv2
import numpy as np
import math  # --- NEW ---
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

        # --- Shape Detection Setup ---
        self.shape_images = {}
        # We will load all 4 shape images
        SHAPE_NAMES = ["heart", "square", "triangle", "house"]

        script_dir = os.path.dirname(os.path.abspath(__file__))

        for shape_name in SHAPE_NAMES:
            try:
                # Look for the PNGs in the *same directory* as this script
                img_path = os.path.join(script_dir, f"{shape_name}.png") 
                img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                if img is None:
                    raise FileNotFoundError(f"File not found: {img_path}")
                
                # We just load the image, no Hu Moments calculation
                self.shape_images[shape_name] = img
                print(f"Loaded replacement image: {shape_name}")
                    
            except Exception as e:
                print(f"ERROR: Could not load replacement image '{shape_name}'. {e}")
                print("       Make sure PNG files are in the 'Drawcode' directory.")

        # --- Load SVM Model and Scaler ---
        try:
            self.svm_model = joblib.load('svm_model.pkl')
            self.scaler = joblib.load('scaler.pkl')
            self.class_names = joblib.load('class_names.pkl')
            print("Successfully loaded SVM model, scaler, and class names.")
        except FileNotFoundError:
            print("ERROR: Model files not found (svm_model.pkl, scaler.pkl, class_names.pkl).")
            print("       Please run train_model.py first.")
            self.svm_model = None # Set to None so we can check later

        # --- State ---
        self.boards = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(self.NUM_BOARDS)]
        self.current_board_index = 0
        self.mode = 'draw'
        self.drawing = False
        self.last_point = None
        self.drawing_source = None # <-- CHANGE 1: Add drawing source
        self.switch_mode = False
        self.cursor_pos = None

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
    #  UNIFIED INPUT HANDLER
    # ============================================================
    def handle_input_event(self, event):
        """
        event = {
            "type": "down" | "move" | "up" | "right_down" | "right_up",
            "x": int,
            "y": int,
            "source": "mouse" | "gesture" | "pen" | "hand" | "face"
        }
        """
        if self.switch_mode:
            return

        x, y = event["x"], event["y"]
        event_type = event["type"]
        board = self.boards[self.current_board_index]
        
        # Always update cursor position from any event
        self.cursor_pos = (x, y) 

        # --- Stamp in manipulation mode ---
        if self.manip_mode and self.manip_img is not None and event_type == "down":
            self._stamp_manip_img()
            return

        # --- Move selection (manipulation dragging) ---
        if self.manip_mode and self.manip_img is not None:
            self._handle_manip_move_from_event(event)
            return

        # --- Copy / Paste Mode ---
        if self.copy_mode:
            # Check for a paste event (right-click) first
            if event["type"] == "right_down" and self.clipboard_img is not None:
                self._handle_paste_clipboard_from_event(event, board)
            
            # Otherwise, handle selection drawing (left-click down, move, up)
            else:
                self._handle_copy_selection_from_event(event)
            
            return # Event is handled, stop processing

        # --- Manipulation selection creation ---
        if self.manip_mode and self.manip_img is None:
            self._handle_manip_selection_from_event(event)
            return

        # --- Drawing / Erasing ---
        self._handle_draw_erase_from_event(event, board)


    # ============================================================
    #  MOUSE WRAPPER -> UNIFIED EVENTS
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
    #  DRAW & ERASE (UNIFIED)
    # ============================================================
    def _handle_draw_erase_from_event(self, event, board):
        x, y = event["x"], event["y"]
        t = event["type"]
        src = event["source"] 

        if t == "down":
            self.drawing = True
            self.drawing_source = src 
            self.last_point = (x, y)
        elif t == "move" and self.drawing:
            # --- Only draw if the event source is the one that started it ---
            if src == self.drawing_source: 
                if self.mode == 'draw':
                    cv2.line(board, self.last_point, (x, y), self.DRAW_COLOR, self.DRAW_RADIUS)
                elif self.mode == 'erase':
                    cv2.line(board, self.last_point, (x, y), self.ERASE_COLOR, self.ERASE_RADIUS)
                self.last_point = (x, y)
        elif t == "up":
            # --- Only lift the pen if the source matches ---
            if src == self.drawing_source:
                # --- CAPTURE CURRENT MODE ---
                mode_when_drawing_started = self.mode 
                
                self.drawing = False
                self.drawing_source = None 
                self.last_point = None
                
                # --- Check for shapes ---
                # Only check if we were in 'draw' mode (not 'erase' mode)
                if mode_when_drawing_started == 'draw':
                    self._check_for_shapes(board)

    # ============================================================
    #  COPY SELECTION (UNIFIED)
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
    #  MANIPULATION (UNIFIED)
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
        # Note: This handler is for right-clicks, so it's fine
        # We could add source-checking here too for safety, but
        # it's less critical as it's a different mouse button.
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
    #  MAIN LOOP
    # ============================================================

    def render_overlay(self, frame):
        overlay = self.boards[self.current_board_index].copy()
        self._draw_copy_highlight(overlay)
        self._draw_manip_highlight(overlay)
        combined = cv2.addWeighted(frame, 1 - self.ALPHA, overlay, self.ALPHA, 0)
        self._draw_preview(combined)
        self._draw_mode_text(combined)

        if self.cursor_pos:
            # If drawing, use the drawing color. If not, use white.
            cursor_color = self.DRAW_COLOR if self.drawing else (255, 255, 255)
            cv2.circle(combined, self.cursor_pos, 5, cursor_color, -1)
            cv2.circle(combined, self.cursor_pos, 5, (0,0,0), 1)
            
        return combined

    # ============================================================
    #  UI DRAWING HELPERS (UNCHANGED)
    # ============================================================
    def _draw_mode_text(self, img):
        mode_str = self.mode.upper() if not self.switch_mode else "PAGE-MODE (Q/W)"
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
    #  KEY INPUT HANDLER (UNCHANGED)
    # ============================================================
    def _handle_key_input(self, key):
        if key == 27:  # ESC
            return False

        if self.switch_mode:
            self._handle_page_switch_keys(key)
            return True

        if key == ord('d'):
            self._stamp_manip_img()
            self._reset_modes(draw=True)

        elif key == ord('e'):
            self._stamp_manip_img()
            self._reset_modes(erase=True)

        elif key == ord('p'):
            self._stamp_manip_img()
            self._reset_modes(page_mode=True)

        elif key == ord('c'):
            self._stamp_manip_img()
            if self.copy_mode:
                self._reset_modes(draw=True)
            else:
                self._reset_modes()
                self.copy_mode = True
                self.mode = 'copy'

        elif key == ord('m'):
            self._stamp_manip_img()
            if self.manip_mode:
                self._reset_modes(draw=True)
            else:
                self._reset_modes()
                self.manip_mode = True
                self.mode = 'manipulate'

        elif key == ord('s'):
            if self.copy_mode:
                self.finalize_and_copy_selection()
            elif self.manip_mode and self.manip_img is None:
                self.finalize_manip_selection()

        elif key == 32:  # SPACE to clear current board
            self.boards[self.current_board_index][:] = 0

        return True


    # ============================================================
    #  SHAPE DETECTION 
    # ============================================================

    @staticmethod
    def _overlay_transparent_image(background, overlay, x, y):
        """Overlays a transparent PNG onto a background image in-place."""
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
        """Calculates log-transformed Hu Moments for a contour."""
        moments = cv2.moments(contour)
        hu_moments = cv2.HuMoments(moments)
        
        # Log-transform hu moments
        for i in range(7):
            hu_moments[i] = -1 * math.copysign(1.0, hu_moments[i][0]) * math.log10(abs(hu_moments[i][0]) + 1e-9)
            
        return hu_moments.flatten()
    
    def _check_for_shapes(self, board):
        """Finds contours, predicts them with the SVM, and replaces them."""
        # Don't try to predict if the model failed to load
        if self.svm_model is None or not self.shape_images:
            return

        print("\n--- Checking for shapes (SVM Method) ---")
        canvas_gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(canvas_gray, 10, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        replacements = []

        for contour in contours:
            # 1. Filter out small noise
            if cv2.contourArea(contour) < 1500:
                continue
                
            # 2. Get features
            features = self._get_features_from_contour(contour)
            features_reshaped = features.reshape(1, -1)
            
            # 3. Scale features
            scaled_features = self.scaler.transform(features_reshaped)
            
            # 4. Predict
            prediction_proba = self.svm_model.predict_proba(scaled_features)[0]
            confidence = prediction_proba.max()
            prediction_index = prediction_proba.argmax()
            
            # 5. Check confidence
            CONF_THRESHOLD = 0.50 # 80% confident
            shape_name = self.class_names[prediction_index] # <-- MOVED HERE

            if confidence >= CONF_THRESHOLD:
                print(f"  -> Matched {shape_name.upper()} (Confidence: {confidence*100:.1f}%)")
                
                if shape_name in self.shape_images:
                    replacements.append((contour, shape_name))
            else:
                 # This line will now work correctly
                 print(f"  -> Low confidence match: {shape_name.upper()} (Confidence: {confidence*100:.1f}%)")

        
        # Apply all replacements
        # (This part is identical to your old code, which is good!)
        for contour, shape_name in replacements:
            x, y, w_drawn, h_drawn = cv2.boundingRect(contour)
            
            # 1. Erase the drawn contour
            cv2.drawContours(board, [contour], -1, self.ERASE_COLOR, cv2.FILLED)
            
            # 2. Get the correct replacement image
            replacement_img = self.shape_images[shape_name]
            
            # 3. Resize and overlay the replacement PNG
            padding_factor = 1.2 
            new_w = int(w_drawn * padding_factor)
            new_h = int(h_drawn * padding_factor)
            
            if new_w > 0 and new_h > 0:
                resized_img = cv2.resize(replacement_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                overlay_x = x - (new_w - w_drawn) // 2
                overlay_y = y - (new_h - h_drawn) // 2
                self._overlay_transparent_image(board, resized_img, overlay_x, overlay_y)

    # ============================================================
    #  MODE MANAGEMENT
    # ============================================================

    def _reset_modes(self, draw=False, erase=False, page_mode=False):
        # Clear all mode flags and selections
        self.copy_mode = False
        self.manip_mode = False
        self.switch_mode = False

        self.selection_points.clear()
        self.manip_selection.clear()

        self.clipboard_img = None
        self.manip_img = None
        self.manip_mask = None
        self.moving = False
        self.move_offset = (0, 0)
        self.drawing = False
        self.drawing_source = None # <-- CHANGE 5: Reset the source
        self.last_point = None
        self.manip_pos = None

        if draw:
            self.mode = 'draw'
        elif erase:
            self.mode = 'erase'
        elif page_mode:
            self.switch_mode = True
            self.mode = 'page-mode'
        else:
            self.mode = 'draw'


    def _handle_page_switch_keys(self, key):
        if key == ord('q'):
            self.current_board_index = (self.current_board_index - 1) % self.NUM_BOARDS
        elif key == ord('w'):
            self.current_board_index = (self.current_board_index + 1) % self.NUM_BOARDS
        elif key == ord('p'):
            self.switch_mode = False