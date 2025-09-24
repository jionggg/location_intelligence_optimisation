import numpy as np
from create_anchor_edges import ANCHORS

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

def create_relative_measurement(anchor_id: int, phone_node_id: str, local_vector: np.ndarray) -> tuple:
    """
    Create a relative measurement (edge) between an anchor and phone node.

    Args:
        anchor_id (int): ID of the anchor (0-3)
        phone_node_id (str): Identifier for the phone pose node (e.g., 'phone_t1')
        local_vector (np.ndarray): Local vector in anchor's coordinate system (x, y, z) in cm

    Returns:
        tuple: (anchor_node, phone_node, displacement_vector) where displacement_vector
               is the relative displacement from anchor to phone in global coordinates

    Raises:
        ValueError: If anchor_id is not in range 0-3 or local_vector is not 3D
    """
    if anchor_id not in ANCHORS:
        raise ValueError(f"Invalid anchor_id: {anchor_id}. Must be 0-3.")

    if local_vector.shape != (3,):
        raise ValueError(f"local_vector must be shape (3,), got {local_vector.shape}")

    # Rotate from local anchor coordinates to global XY
    v_global = ANCHOR_R[anchor_id] @ local_vector

    # Return relative measurement: (anchor_node, phone_node, displacement)
    anchor_node = f"anchor_{anchor_id}"
    return (anchor_node, phone_node_id, v_global)


# Example usage
if __name__ == "__main__":
    # Example: create relative measurement from anchor 3 to phone at time 1
    local_vec = np.array([10.0, 20.0, 5.0])  # cm
    relative_edge = create_relative_measurement(3, "phone_t1", local_vec)
    print(f"Local vector from anchor 3: {local_vec}")
    print(f"Relative measurement: {relative_edge}")

    # Test all anchors
    print("\nAnchor-phone relative measurements:")
    for anchor_id in range(4):
        relative_edge = create_relative_measurement(anchor_id, "phone_t1", local_vec)
        print(f"Anchor {anchor_id}: {relative_edge}")
