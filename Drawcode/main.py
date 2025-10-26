import cv2
from arboard import ARDrawingBoard
from gestures import FaceHandTracker

def main():
    # 1. Initialize the drawing board
    # We use camera_index=0 by default
    board = ARDrawingBoard(camera_index=0)

    # 2. Initialize the tracker and link it to the board
    # We pass the board's input handler as the tracker's callback.
    # Now, whenever the tracker detects a gesture, it will call board.handle_input_event()
    tracker = FaceHandTracker(event_callback=board.handle_input_event)

    # Check if camera opened successfully
    if not board.cap.isOpened():
        print("Cannot open camera")
        return

    print("--- AR Drawing Board with Hand Tracking ---")
    print("Point your index finger to draw.")
    print("Show an open palm to move the cursor (hover).")
    print("Make a fist to stop drawing.")
    print("\nKeyboard Controls:")
    print("d/draw, e/erase, p/page-mode (Q/W), c/copy, m/manipulate, s/save selection, SPACE/clear, ESC/quit")


    # 3. Main application loop
    while True:
        # --- Read Frame ---
        # We get the frame from the board's camera object
        ret, frame = board.cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        # Flip the frame so it's a mirror image
        frame = cv2.flip(frame, 1)
        
        # --- Process Tracking ---
        # Pass the raw camera frame to the tracker.
        # It will find hands, draw landmarks, and most importantly,
        # call board.handle_input_event() if it recognizes a drawing gesture.
        # It returns the frame with the hand landmarks drawn on it.
        frame_with_hands = tracker.process_frame(frame)

        # --- Render Board ---
        # Pass the frame (now with hand landmarks) to the board's renderer.
        # It will overlay the transparent drawing board, UI, and cursor.
        combined_frame = board.render_overlay(frame_with_hands)

        # --- Display ---
        cv2.imshow("AR Drawing Board", combined_frame)

        # --- Handle Keyboard Input ---
        # Check for keyboard presses for changing modes, etc.
        key = cv2.waitKey(10) & 0xFF
        if not board._handle_key_input(key):
            break # Exit loop if ESC is pressed

    # 4. Cleanup
    board.cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()