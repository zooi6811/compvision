import cv2
import cv2.aruco as aruco
import numpy as np
from PIL import Image

# --- CAMERA SETUP ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Camera calibration (replace with your own!)
camera_matrix = np.array([[800, 0, 320],
                          [0, 800, 240],
                          [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros((5, 1))

# --- ARUCO SETUP ---
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)
marker_length = 0.05  # meters

# --- LOAD OBJ + MTL + TEXTURE ---
def load_obj_with_mtl_texture(obj_file, mtl_file):
    vertices = []
    faces = []
    face_colors = []

    material_colors = {}
    material_textures = {}

    # --- Parse MTL ---
    with open(mtl_file, 'r') as f:
        current_mtl = None
        for line in f:
            if line.startswith("newmtl"):
                current_mtl = line.strip().split()[1]
            elif line.startswith("Kd") and current_mtl:
                kd = [float(x) for x in line.strip().split()[1:]]
                color = (int(kd[2]*255), int(kd[1]*255), int(kd[0]*255))
                material_colors[current_mtl] = color
            elif line.startswith("map_Kd") and current_mtl:
                texture_file = line.strip().split()[1]
                img = Image.open(texture_file).convert("RGB")
                material_textures[current_mtl] = np.array(img)

    # --- Parse OBJ ---
    current_mtl = None
    with open(obj_file, 'r') as f:
        for line in f:
            if line.startswith('v '):
                vertices.append([float(x) for x in line.strip().split()[1:]])
            elif line.startswith('usemtl'):
                current_mtl = line.strip().split()[1]
            elif line.startswith('f '):
                idxs = [int(x.split('/')[0])-1 for x in line.strip().split()[1:]]
                faces.append(idxs)

                if current_mtl in material_textures:
                    tex = material_textures[current_mtl]
                    h, w, _ = tex.shape
                    color = tuple(tex[h//2, w//2][::-1])  # BGR
                else:
                    color = material_colors.get(current_mtl, (0,255,0))

                face_colors.append(color)

    return np.array(vertices, dtype=np.float32), faces, face_colors

vertices, faces, face_colors = load_obj_with_mtl_texture("model.obj", "model.mtl")

# --- SCALE AND CENTER MODEL ---
def normalize_model(vertices, marker_length):
    vertices = vertices - np.mean(vertices, axis=0)
    scale = marker_length / (np.max(vertices[:, :2]) - np.min(vertices[:, :2]))
    vertices *= scale
    return vertices

vertices = normalize_model(vertices, marker_length)

# --- ROTATE MODEL 90° AROUND Y-AXIS ---
def rotate_model_y(vertices, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    R = np.array([[ np.cos(angle_rad), 0, np.sin(angle_rad)],
                  [ 0,               1, 0              ],
                  [-np.sin(angle_rad),0, np.cos(angle_rad)]], dtype=np.float32)
    return vertices @ R.T

vertices = rotate_model_y(vertices, 90)

# --- DRAW AXIS ---
def draw_axis(img, camera_matrix, dist_coeffs, rvec, tvec, length=0.03):
    axis_points = np.float32([[length,0,0],[0,length,0],[0,0,length]])
    origin_pts = np.float32([[0,0,0]])
    img_origin,_ = cv2.projectPoints(origin_pts, rvec, tvec, camera_matrix, dist_coeffs)
    img_pts,_ = cv2.projectPoints(axis_points, rvec, tvec, camera_matrix, dist_coeffs)
    origin = tuple(img_origin.ravel().astype(int))
    x_axis = tuple(img_pts[0].ravel().astype(int))
    y_axis = tuple(img_pts[1].ravel().astype(int))
    z_axis = tuple(img_pts[2].ravel().astype(int))
    cv2.line(img, origin, x_axis, (0,0,255), 2)
    cv2.line(img, origin, y_axis, (0,255,0), 2)
    cv2.line(img, origin, z_axis, (255,0,0), 2)

# --- MAIN LOOP ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is not None:
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id == 0:
                c = corners[i].reshape((4,2)).astype(np.float32)
                half_size = marker_length / 2
                obj_pts = np.array([[-half_size, half_size, 0],
                                    [half_size, half_size, 0],
                                    [half_size, -half_size, 0],
                                    [-half_size, -half_size, 0]], dtype=np.float32)

                retval, rvec, tvec = cv2.solvePnP(obj_pts, c, camera_matrix, dist_coeffs)

                draw_axis(frame, camera_matrix, dist_coeffs, rvec, tvec)

                # --- PROJECT 3D MODEL ---
                img_pts, _ = cv2.projectPoints(vertices, rvec, tvec, camera_matrix, dist_coeffs)
                img_pts = img_pts.reshape(-1,2).astype(int)

                # Draw faces with texture colors
                for f_idx, face in enumerate(faces):
                    color = face_colors[f_idx]
                    pts = np.array([img_pts[idx] for idx in face], np.int32)
                    cv2.fillPoly(frame, [pts], color)

    cv2.imshow("ARUCO 3D Model with Texture", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
