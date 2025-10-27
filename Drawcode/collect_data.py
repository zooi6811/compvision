import cv2
import numpy as np
import os
import time

# --- Configuration ---
# Match these to your shape names
SHAPES = {
    'h': 'heart',
    's': 'square',
    't': 'triangle',
    'o': 'house'  # 'o' for 'hOuse' since 'h' is taken
}
DATA_DIR = 'data'
CANVAS_SIZE = (480, 640, 3)
DRAW_COLOR = (255, 255, 255) # Draw in white on black bg
DRAW_RADIUS = 5

# --- State ---
canvas = np.zeros(CANVAS_SIZE, dtype=np.uint8)
drawing = False
last_point = None

# --- Mouse Callback ---
def mouse_callback(event, x, y, flags, param):
    global drawing, last_point, canvas
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        last_point = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.line(canvas, last_point, (x, y), DRAW_COLOR, DRAW_RADIUS)
            last_point = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        last_point = None

# --- Main ---
cv2.namedWindow("Data Collector")
cv2.setMouseCallback("Data Collector", mouse_callback)

print("--- Shape Data Collector ---")
print("Instructions:")
print("1. Draw a single shape on the black canvas.")
print("2. Press the key corresponding to the shape to save it:")
for key, name in SHAPES.items():
    print(f"   Press '{key}' to save a '{name}'")
print("\nPress 'c' to CLEAR the canvas.")
print("Press 'ESC' to QUIT.")

while True:
    cv2.imshow("Data Collector", canvas)
    key = cv2.waitKey(1) & 0xFF

    if key == 27: # ESC
        break
    elif key == ord('c'): # Clear
        canvas[:] = 0
        print("Canvas cleared.")
    elif chr(key) in SHAPES:
        shape_name = SHAPES[chr(key)]
        shape_dir = os.path.join(DATA_DIR, shape_name)
        
        if not os.path.exists(shape_dir):
            print(f"Error: Directory not found: {shape_dir}")
            continue

        # Find the contour of the drawing
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour
            contour = max(contours, key=cv2.contourArea)
            
            # Get bounding box and save the ROI
            x, y, w, h = cv2.boundingRect(contour)
            if w > 0 and h > 0:
                roi = gray[y:y+h, x:x+w]
                
                # Create a unique filename
                filename = f"{shape_name}_{int(time.time() * 1000)}.png"
                save_path = os.path.join(shape_dir, filename)
                cv2.imwrite(save_path, roi)
                
                print(f"Saved: {save_path}")
                canvas[:] = 0 # Clear canvas after saving
            else:
                print("No drawing found to save.")
        else:
            print("No drawing found to save.")

cv2.destroyAllWindows()