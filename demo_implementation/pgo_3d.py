"""
pgo_3d.py

3D Pose Graph Optimization for real-time UWB localization.
Extends pgo_test.py to support 3D positions, string node names, and anchoring.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy.optimize import least_squares


def solve_pose_graph_3d(graph_data: Dict, anchored_nodes: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, np.ndarray]:
    """
    Solve 3D pose graph optimization problem.

    Args:
        graph_data: Dict with 'nodes' and 'edges' from DataIngestor
        anchored_nodes: Optional dict of node_id -> fixed_position for anchoring

    Returns:
        Dict of optimized node positions {node_id: np.array([x, y, z])}
    """
    nodes = graph_data['nodes']
    edges = graph_data['edges']

    # Extract node information
    node_ids = list(nodes.keys())
    node_indices = {node_id: idx for idx, node_id in enumerate(node_ids)}
    n = len(node_ids)

    print(f"PGO: Optimizing {n} nodes with {len(edges)} edges")

    # Create initial guess (zeros, will be adjusted for anchored nodes)
    initial_guess = np.zeros((n, 3))

    # Set anchored nodes to their fixed positions
    anchored_indices = []
    if anchored_nodes:
        for node_id, fixed_pos in anchored_nodes.items():
            if node_id in node_indices:
                idx = node_indices[node_id]
                initial_guess[idx] = fixed_pos
                anchored_indices.append(idx)
                print(f"PGO: Anchoring {node_id} to {fixed_pos}")

    # Flatten initial guess
    x0 = initial_guess.flatten()

    # Residual function
    def residuals(flat_positions):
        positions = flat_positions.reshape((n, 3))
        res = []

        for edge in edges:
            node_i, node_j, measured_disp = edge

            # Skip if either node not in our optimization
            if node_i not in node_indices or node_j not in node_indices:
                continue

            idx_i = node_indices[node_i]
            idx_j = node_indices[node_j]

            # Estimated displacement
            est_disp = positions[idx_j] - positions[idx_i]

            # Residual
            err = est_disp - measured_disp
            res.append(err)

        if not res:
            return np.array([])

        return np.concatenate(res)

    # Anchored residual function (fixes anchored nodes)
    def anchored_residuals(flat_pos):
        pos = flat_pos.reshape((n, 3))

        # Force anchored nodes to their fixed positions
        for idx in anchored_indices:
            if anchored_nodes:
                # Find which node this index corresponds to
                for node_id, fixed_pos in anchored_nodes.items():
                    if node_indices[node_id] == idx:
                        pos[idx] = fixed_pos
                        break

        return residuals(pos.flatten())

    # Optimize
    try:
        result = least_squares(anchored_residuals, x0, method='trf', max_nfev=1000)
        optimized_flat = result.x
        optimized_positions = optimized_flat.reshape((n, 3))

        # Create result dictionary
        result_dict = {}
        for node_id, idx in node_indices.items():
            result_dict[node_id] = optimized_positions[idx].copy()

        print(f"PGO: Optimization successful. Final cost: {result.cost:.6f}")
        return result_dict

    except Exception as e:
        print(f"PGO: Optimization failed: {e}")
        # Return initial guess as fallback
        fallback_result = {}
        for node_id, idx in node_indices.items():
            fallback_result[node_id] = initial_guess[idx].copy()
        return fallback_result


def create_anchor_anchoring(anchor_3_pos: np.ndarray, anchor_0_pos: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Create anchoring configuration for PGO.
    Fixes anchor_3 at origin and anchor_2 at its known position to prevent rotation.

    Args:
        anchor_3_pos: Known position of anchor_3 (will be mapped to origin)
        anchor_0_pos: Known position of anchor_0

    Returns:
        Dict with anchor_3 fixed at origin and anchor_2 at its known position
    """
    # Anchor both bottom anchors to fix rotation and translation
    # anchor_3 at origin (0,0,0)
    # anchor_2 at bottom-right (ROOM_W, 0, 0) = (440, 0, 0)

    from create_anchor_edges import ANCHORS
    anchor_2_pos = ANCHORS[2]  # (440, 0, 0)

    return {
        'anchor_3': np.array([0.0, 0.0, 0.0]),     # Fix translation (origin)
        'anchor_2': anchor_2_pos                      # Fix rotation (bottom-right corner)
    }


# Example usage and testing
if __name__ == "__main__":
    from ingest_data import DataIngestor
    from create_anchor_edges import ANCHORS

    print("Testing 3D PGO...")

    # Create ingestor and add some test data
    ingestor = DataIngestor(window_size_seconds=1.0)

    # Add test measurements
    import time
    current_time = time.time()  # Use actual current time
    test_measurements = [
        (current_time - 0.9, 0, np.array([10.0, 20.0, 5.0])),  # Within 1-second window
        (current_time - 0.8, 1, np.array([15.0, 25.0, 6.0])),
        (current_time - 0.7, 2, np.array([12.0, 22.0, 4.0])),
        (current_time - 0.6, 3, np.array([8.0, 18.0, 7.0])),
    ]

    for timestamp, anchor_id, local_vec in test_measurements:
        ingestor.add_measurement(timestamp, anchor_id, local_vec)

    # Get graph data
    graph_data = ingestor.get_latest_graph_data()

    if graph_data:
        print(f"Graph has {len(graph_data['edges'])} edges")

        # Create anchoring (fix anchor_3 to origin for gauge freedom)
        anchoring = create_anchor_anchoring(ANCHORS[3], ANCHORS[0])

        # Solve PGO
        optimized_positions = solve_pose_graph_3d(graph_data, anchoring)

        print("\nOptimized positions:")
        for node_id, pos in optimized_positions.items():
            print(f"  {node_id}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")

        # Apply anchoring transformation to get final global coordinates
        from ingest_data import apply_anchoring_transformation, extract_phone_position

        anchored_positions =  apply_anchoring_transformation(optimized_positions)

        print("\nAfter anchoring transformation:")
        for node_id, pos in anchored_positions.items():
            print(f"  {node_id}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")

        # Extract phone position
        phone_node_id = graph_data['binned_data'].phone_node_id
        x, y, z = extract_phone_position(anchored_positions, phone_node_id)
        print(f"\nFinal phone position: ({x:.3f}, {y:.3f}, {z:.3f}) cm")

    else:
        print("No graph data available for testing")
