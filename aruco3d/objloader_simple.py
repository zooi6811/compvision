import numpy as np

class OBJ:
    """
    Simple OBJ loader to use with OpenCV rendering.
    Supports vertices and faces only (triangles/quads).
    """
    def __init__(self, filename, swapyz=False, scale=1.0):
        self.vertices = []
        self.faces = []
        self.scale = scale

        try:
            with open(filename, 'r') as f:
                for line in f:
                    if line.startswith('#') or len(line.strip()) == 0:
                        continue
                    values = line.strip().split()
                    if values[0] == 'v':  # vertex
                        x, y, z = map(float, values[1:4])
                        if swapyz:
                            x, y, z = x, z, y
                        self.vertices.append([x*scale, y*scale, z*scale])
                    elif values[0] == 'f':  # face
                        face = []
                        for v in values[1:]:
                            # faces can be like "f 1 2 3" or "f 1/1/1 2/2/2 3/3/3"
                            w = v.split('/')[0]
                            face.append(int(w))
                        self.faces.append((face,))
            self.vertices = np.array(self.vertices, dtype=np.float32)
        except FileNotFoundError:
            print(f"ERROR: OBJ file {filename} not found")
            raise
