import cv2
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import math

DATA_DIR = 'data'

def get_features_from_image(img_path):
    """Loads an image, finds its contour, and returns Hu Moments."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    
    # Threshold and find contours
    _, thresh = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
        
    # Use the largest contour
    contour = max(contours, key=cv2.contourArea)
    
    # Calculate moments
    moments = cv2.moments(contour)
    
    # Calculate Hu Moments
    hu_moments = cv2.HuMoments(moments)
    
    # Log-transform hu moments for better scale invariance
    for i in range(7):
        hu_moments[i] = -1 * math.copysign(1.0, hu_moments[i]) * math.log10(abs(hu_moments[i]) + 1e-9) # Add epsilon
        
    return hu_moments.flatten()

# --- Main Training ---
print("Starting model training...")

features_list = []
labels_list = []
class_names = [] # This will be our list of names, e.g., ['heart', 'house', 'square']

for shape_name in sorted(os.listdir(DATA_DIR)):
    shape_dir = os.path.join(DATA_DIR, shape_name)
    if not os.path.isdir(shape_dir):
        continue
        
    print(f"Processing shape: {shape_name}")
    class_names.append(shape_name)
    label_index = len(class_names) - 1 # e.g., 0 for 'heart', 1 for 'house'
    
    for img_file in os.listdir(shape_dir):
        img_path = os.path.join(shape_dir, img_file)
        features = get_features_from_image(img_path)
        
        if features is not None:
            features_list.append(features)
            labels_list.append(label_index)

if not features_list:
    print("Error: No data found. Did you run collect_data.py?")
    exit()

print(f"\nFound {len(features_list)} samples.")
print(f"Classes: {class_names}")

X = np.array(features_list)
y = np.array(labels_list)

# 1. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 2. Scale features (CRITICAL for SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Training SVM classifier...")
# We use probability=True to get confidence scores later
svm_model = SVC(kernel='rbf', C=10, gamma=0.1, probability=True) 
svm_model.fit(X_train_scaled, y_train)

# 4. Test the model
y_pred = svm_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy on Test Set: {accuracy * 100:.2f}%")

# 5. Save the artifacts
joblib.dump(svm_model, 'svm_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(class_names, 'class_names.pkl') # Save the class name list

print("\nSuccessfully saved:")
print("- svm_model.pkl (The trained SVM)")
print("- scaler.pkl (The feature scaler)")
print("- class_names.pkl (The list of shape names)")