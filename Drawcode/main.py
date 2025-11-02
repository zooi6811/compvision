import cv2
import time
import numpy as np 
from arboard import ARDrawingBoard      
from gestures import FaceHandTracker
from mitchtags import MarkerTracker  
              

def main():
    # 1. Initialize
    board = ARDrawingBoard(camera_index=0)
    tracker_hands = FaceHandTracker(board=board)
    tracker_tags = MarkerTracker()

    if not board.cap.isOpened():
        print("Cannot open camera")
        return

    # Get the frame width for coordinate correction
    frame_w = int(board.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    print("--- AR Drawing Board with Mixed Coordinate Systems ---")
    print("Camera feed and hand tracking are flipped (mirror view).")
    print("Marker detection runs on the unflipped original frame.")
    
    # --- Main application loop ---
    while True:
        current_time = time.time()
        
        ret, frame_original = board.cap.read()
        if not ret:
            time.sleep(0.01)
            continue
        
        # 1. Create the FLIPPED frame (for user viewing and hand tracking)
        frame_flipped = cv2.flip(frame_original, 1)

        # 2. Detect custom tags on the **ORIGINAL, UNFLIPPED** frame
        # We pass the original frame to the marker tracker.
        tracked_markers_list, frame_with_tags_debug = tracker_tags.process_frame(frame_original, current_time)
        
        # We start with the flipped frame for drawing/hand tracking
        current_drawing_frame = frame_flipped.copy() 

        # 3. Data Conversion & Coordinate Translation
        detected_tags_boxes = {}
        
        for marker in tracked_markers_list:
            marker_id = marker['id']
            corners_original = marker['corners'] 
            
            # Calculate the bounding box in the **ORIGINAL** coordinate system
            x_orig, y_orig, w, h = cv2.boundingRect(corners_original.astype(int))
            
            # 💡 KEY STEP: Translate the X coordinate to the FLIPPED system
            # The new X starts where the old X+W finished, mirrored across the center.
            x_flipped = frame_w - (x_orig + w)
            y_flipped = y_orig # Y-coordinate remains the same
            
            # Store the box in the FLIPPED coordinate system
            detected_tags_boxes[marker_id] = (x_flipped, y_flipped, w, h)
            
            # 4. Draw Bounding Boxes (on the FLIPPED frame)
            if marker_id <= 10:
                # Draw on the flipped frame using the translated coordinates
                cv2.rectangle(current_drawing_frame, (x_flipped, y_flipped), 
                              (x_flipped + w, y_flipped + h), (255, 0, 255), 2)
                cv2.putText(current_drawing_frame, f"ID: {marker_id}", (x_flipped, y_flipped - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

        # 5. Process for hands (on the FLIPPED frame)
        # NOTE: Hand tracker logic must be correct for the flipped frame.
        # This means your FaceHandTracker's left/right logic (in gestures.py) 
        # MUST be swapped back to its **original, unflipped logic** because
        # the hand tracker's input frame is now visually flipped!
        frame_with_hands = tracker_hands.process_frame(current_drawing_frame)

        # 6. Check for Tag Interactions (all coordinates are in the FLIPPED space)
        board.check_tag_interaction(detected_tags_boxes)

        # 7. Render the final board overlay
        combined_frame = board.render_overlay(frame_with_hands)

        # --- Display ---
        cv2.imshow("AR Drawing Board", combined_frame)

        # --- Handle Keyboard Input ---
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('d'):
            # This will show the debug on the *original* frame if you implement it
            custom_tracker.DEBUG = not custom_tracker.DEBUG
            print(f"\nSet custom_tracker.DEBUG to {custom_tracker.DEBUG}")

        elif not board._handle_key_input(key):
            break

    # 8. Cleanup
    board.cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()