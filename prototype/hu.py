import cv2
import numpy as np
import os
import math

def get_hu_moments(image):
    """
    Calculates Hu Moments for a 3-channel, light-on-dark image.
    ASSUMES:
    1. Image is 3-channel BGR (no transparency).
    2. Shape is white (or light-colored).
    3. Background is black.
    """
    if image is None:
        print("  [Error] Image is None.")
        return None, None
        
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    if cv2.countNonZero(thresh) < 100:
         print("  [Error] No significant pixels/shape found.")
         return None, None

    moments = cv2.moments(thresh, binaryImage=True)
    hu_moments = cv2.HuMoments(moments)
    
    return hu_moments, None

def compare_shapes(test_hu, golden_templates):
    """
    Compares a test shape's Hu Moments using Pearson Correlation.
    A score closer to 1.0 is a better match.
    """
    if test_hu is None:
        print("  [Error] Could not find a valid contour in the test image.")
        return

    test_hu_flat = test_hu.flatten()

    best_match = None
    best_score = -float('inf') # We want to MAXIMIZE correlation

    print("  --- Comparison Scores (Pearson Correlation) ---")
    for name, golden_hu in golden_templates.items():
        if golden_hu is None:
            continue
        
        golden_hu_flat = golden_hu.flatten()
            
        # Calculate the 2x2 correlation matrix
        corr_matrix = np.corrcoef(test_hu_flat, golden_hu_flat)
        
        # The correlation is at [0, 1] (or [1, 0])
        score = corr_matrix[0, 1]

        # Handle potential NaN scores if one vector has zero variance
        # (e.g., all moments are zero)
        if np.isnan(score):
            # If both are identical (e.g., all zeros), it's a perfect match
            if np.allclose(test_hu_flat, golden_hu_flat):
                score = 1.0
            else: # Otherwise, they are a bad match
                score = -1.0

        print(f"  Score vs '{name.upper()}': {score:.6f}")
        
        if score > best_score:
            best_score = score
            best_match = name
            
    print(f"  --------------------------")
    print(f"  🏆 Best Match: {best_match.upper()} (Score: {best_score:.6f})")


def main():
    print("====== Hu Moments Shape Identification Demo ======")
    print("This script tests 'light-on-dark' shapes.\n")

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # --- 1. Load Templates and Get Golden Hu Moments ---
    
    GOLDEN_NAMES = ["heart", "square", "triangle", "house"]
    golden_templates = {}

    print("--- Loading Golden Templates (as 3-channel) ---")
    for name in GOLDEN_NAMES:
        img_path = os.path.join(script_dir, f"{name}.png")
        if not os.path.exists(img_path):
            print(f"  [Warning] Golden template not found: {name}.png")
            continue
            
        img = cv2.imread(img_path, cv2.IMREAD_COLOR) 
        golden_hu, _ = get_hu_moments(img) 
        
        if golden_hu is not None:
            golden_templates[name] = golden_hu
            print(f"  ✅ Loaded and analysed '{name}.png'")
        else:
            print(f"  [Error] Could not find contour in '{name}.png'")

    if not golden_templates:
        print("\nERROR: No golden templates were loaded. Exiting.")
        return

    # --- 2. Define Test Cases (Your Drawn Images) ---
    drawn_shape_files = [
        "draw_heart.png",
        "draw_house.png",
        "draw_square.png",
        "draw_triangle.png",
    ]

    # --- 3. Run Comparisons ---
    print("\n====== Running Tests on Drawn Shapes (as 3-channel) ======")
    
    for filename in drawn_shape_files:
        print(f"\n--- TEST: {filename} ---")
        
        img_path = os.path.join(script_dir, filename)
        if not os.path.exists(img_path):
            print(f"  [Skipping] File not found: {filename}")
            continue
            
        drawn_img = cv2.imread(img_path, cv2.IMREAD_COLOR) 
        test_hu, _ = get_hu_moments(drawn_img)
        
        compare_shapes(test_hu, golden_templates)

    print("\n====== Demo Complete ======")


if __name__ == "__main__":
    main()