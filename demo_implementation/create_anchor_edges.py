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


def create_anchor_anchor_edges() -> list:
    """
    Create all 6 anchor-anchor edges with exact known distances (perfect version).

    Returns:
        list: List of tuples (anchor_node_i, anchor_node_j, displacement_vector)
              where displacement_vector is the exact distance from i to j
    """
    edges = []
    anchor_ids = [0, 1, 2, 3]

    for i in anchor_ids:
        for j in anchor_ids:
            if i < j:  # Avoid duplicates (i,j) and (j,i)
                displacement = ANCHORS[j] - ANCHORS[i]  # Vector from i to j
                edges.append((f"anchor_{i}", f"anchor_{j}", displacement))

    return edges


# Example usage
if __name__ == "__main__":
    print("Anchor-anchor edges:")
    anchor_edges = create_anchor_anchor_edges()
    for edge in anchor_edges:
        print(f"{edge[0]} -> {edge[1]}: {edge[2]}")
