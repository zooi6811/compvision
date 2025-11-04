import cv2
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
# --- MODIFIED: Added confusion_matrix and ConfusionMatrixDisplay ---
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import math
import matplotlib.pyplot as plt  # <-- NEW: For plotting the matrix

DATA_DIR = 'data'


def get_features_from_image(img_path):
    """Loads an image, finds its contour, and returns Hu Moments."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    _, thresh = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    moments = cv2.moments(contour)
    hu_moments = cv2.HuMoments(moments)

    # Log-transform hu moments
    for i in range(7):
        # Fix for NumPy deprecation warning
        hu_moments[i] = -1 * math.copysign(1.0, hu_moments[i][0]) * math.log10(abs(hu_moments[i][0]) + 1e-9)

    return hu_moments.flatten()


# --- Main Training ---
print("Starting model training...")

features_list = []
labels_list = []
class_names = []

for shape_name in sorted(os.listdir(DATA_DIR)):
    shape_dir = os.path.join(DATA_DIR, shape_name)
    if not os.path.isdir(shape_dir):
        continue

    print(f"Processing shape: {shape_name}")
    class_names.append(shape_name)
    label_index = len(class_names) - 1

    count = 0  # <-- NEW: Let's count samples
    for img_file in os.listdir(shape_dir):
        img_path = os.path.join(shape_dir, img_file)
        features = get_features_from_image(img_path)

        if features is not None:
            features_list.append(features)
            labels_list.append(label_index)
            count += 1

    print(f"  -> Added {count} samples.")  # <-- NEW

if not features_list:
    print("Error: No data found. Did you run collect_data.py?")
    exit()

print(f"\nFound {len(features_list)} total samples.")
print(f"Classes: {class_names}")

X = np.array(features_list)
y = np.array(labels_list)

# 1. Split data (we still need a final test set)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 2. Scale features
scaler = StandardScaler()
# Fit on the training data ONLY
X_train_scaled = scaler.fit_transform(X_train)
# Transform the test data using the same scaler
X_test_scaled = scaler.transform(X_test)

# --- NEW: Hyperparameter Tuning with GridSearchCV ---
print("\nStarting Hyperparameter Tuning (GridSearchCV)...")
print("This may take a few minutes.")

# Define the 'grid' of parameters to search
# These are good ranges to start with for 'rbf' kernel
param_grid = {
    'C': [0.1, 1, 10, 100, 1000],
    'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
}

# Create the GridSearchCV object
# cv=5 means 5-fold cross-validation
# n_jobs=-1 uses all your CPU cores to speed it up
grid_search = GridSearchCV(
    SVC(kernel='rbf', probability=True),
    param_grid,
    cv=5,
    verbose=2,
    n_jobs=-1
)

# Run the search on the (scaled) training data
grid_search.fit(X_train_scaled, y_train)

# --- End of NEW section ---

# Get the best model found by the grid search
best_svm_model = grid_search.best_estimator_

print("\n--- Tuning Complete ---")
print(f"Best Parameters Found: {grid_search.best_params_}")
print(f"Best Cross-validation Score: {grid_search.best_score_ * 100:.2f}%")

# 4. Test the *best* model on the *unseen* test set
y_pred = best_svm_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nFinal Model Accuracy on Test Set: {accuracy * 100:.2f}%")

print("\n--- Confusion Matrix ---")

# Generate the matrix
cm = confusion_matrix(y_test, y_pred)

# Print the text-based matrix
print(f"Classes: {class_names}")
print("Matrix (Rows=True, Cols=Predicted):\n", cm)


# Plot and save the graphical matrix
try:
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    
    # This creates a pyplot figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))  
    
    # This plots the matrix onto the pyplot axis
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    
    # This uses pyplot to set the title
    plt.title("Confusion Matrix")
    
    output_filename = "confusion_matrix.png"
    # This uses pyplot to save the figure
    plt.savefig(output_filename)
    print(f"\nSaved graphical confusion matrix to: {output_filename}")

    # --- OPTIONAL: Show the plot ---
    # If you also want the plot to pop up in a new window,
    # add this line:
    # plt.show() 
    
except Exception as e:
    print(f"\nCould not save graphical matrix (requires matplotlib): {e}")


# 5. Save the artifacts
# We save the best_svm_model (which is already trained)
joblib.dump(best_svm_model, 'svm_model.pkl')
# We save the scaler that was fit to the training data
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(class_names, 'class_names.pkl')

print("\nSuccessfully saved:")
print("- svm_model.pkl (The *best* trained SVM)")
print("- scaler.pkl (The feature scaler)")
print("- class_names.pkl (The list of shape names)")