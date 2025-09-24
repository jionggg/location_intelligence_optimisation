import numpy as np

# -----------------------------
# Room/grid geometry (cm)
# -----------------------------
SQUARE_CM = 55.0
NX, NY = 8, 10 
ROOM_W, ROOM_H = NX * SQUARE_CM, NY * SQUARE_CM

# Anchor positions in global frame (x right, y up), origin at Anchor 3 (bottom-left)
ANCHORS = {
    0: np.array([ROOM_W, ROOM_H, 0.0]),  # top-right
    1: np.array([0.0,     ROOM_H, 0.0]), # top-left
    2: np.array([ROOM_W,  0.0,    0.0]), # bottom-right
    3: np.array([0.0,     0.0,    0.0]), # bottom-left (origin)
}

# 45° "facing into the room" yaw for each board; local x points along heading into the room,
# local y is left of the board, z is up. Rotation is about +z only.
def Rz(deg: float) -> np.ndarray:
    rad = np.deg2rad(deg); c, s = np.cos(rad), np.sin(rad)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]], dtype=float)

ANCHOR_R = {
    0: Rz(225.0),  # top-right faces bottom-left
    1: Rz(315.0),  # top-left  faces bottom-right
    2: Rz(135.0),  # bottom-right faces top-left
    3: Rz(45.0),   # bottom-left  faces top-right
}

def transform_to_global_vector(anchor_id: int, local_vector: np.ndarray) -> np.ndarray:
    """
    Transform a local vector from an anchor's coordinate system to global coordinates.

    Args:
        anchor_id (int): ID of the anchor (0-3)
        local_vector (np.ndarray): Local vector in anchor's coordinate system (x, y, z) in cm

    Returns:
        np.ndarray: Global position vector (x, y, z) in cm

    Raises:
        ValueError: If anchor_id is not in range 0-3 or local_vector is not 3D
    """
    if anchor_id not in ANCHORS:
        raise ValueError(f"Invalid anchor_id: {anchor_id}. Must be 0-3.")

    if local_vector.shape != (3,):
        raise ValueError(f"local_vector must be shape (3,), got {local_vector.shape}")

    # Rotate from local anchor coordinates to global XY
    v_global = ANCHOR_R[anchor_id] @ local_vector

    # Add anchor's global position
    global_position = ANCHORS[anchor_id] + v_global

    return global_position

# Example usage
if __name__ == "__main__":
    # Example: transform a local vector [10, 20, 5] cm from anchor 3
    local_vec = np.array([10.0, 20.0, 5.0])  # cm
    global_pos = transform_to_global_vector(3, local_vec)
    print(f"Local vector from anchor 3: {local_vec}")
    print(f"Global position: {global_pos}")

    # Test all anchors
    for anchor_id in range(4):
        global_pos = transform_to_global_vector(anchor_id, local_vec)
        print(f"Anchor {anchor_id}: {global_pos}")
