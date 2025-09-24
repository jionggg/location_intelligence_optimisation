from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

def solve_pose_graph(edge_list: List[Dict[Tuple[int, int], Tuple[float, float]]]) -> pd.DataFrame:
    # Step 1: Flatten edge list into (i, j, displacement)
    edges = []
    nodes = set()
    for entry in edge_list:
        for (i, j), (dx, dy) in entry.items():
            displacement = np.array([dx, dy])
            edges.append((i, j, displacement))
            nodes.update([i, j])

    # Step 2: Indexing
    nodes = sorted(list(nodes))
    node_indices = {node: idx for idx, node in enumerate(nodes)}
    n = len(nodes)

    # Step 3: Residual function
    def residuals(flat_positions):
        positions = flat_positions.reshape((n, 2))
        res = []
        for i, j, measured in edges:
            idx_i = node_indices[i]
            idx_j = node_indices[j]
            est_disp = positions[idx_j] - positions[idx_i]
            err = est_disp - measured
            res.append(err)
        return np.concatenate(res)

    # Step 4: Anchor node 0
    x0 = np.zeros((n, 2))
    x0[0] = [0.0, 0.0]
    initial_guess = x0.flatten()

    def anchored_residuals(flat_pos):
        pos = flat_pos.reshape((n, 2))
        pos[0] = [0.0, 0.0]
        return residuals(pos.flatten())

    # Step 5: Optimize
    result = least_squares(anchored_residuals, initial_guess)
    optimized_positions = result.x.reshape((n, 2))

    # Step 6: Output as DataFrame
    position_dict = {node: optimized_positions[idx] for node, idx in node_indices.items()}
    df = pd.DataFrame({
        'Node': list(position_dict.keys()),
        'X': [v[0] for v in position_dict.values()],
        'Y': [v[1] for v in position_dict.values()]
    }).sort_values('Node')

    return df

# ---------------- Visualisation ---------------- #

def visualize_pose_graph(edge_list: List[Dict[Tuple[int, int], Tuple[float, float]]], positions_df: pd.DataFrame, show_measurements: bool = True) -> None:
    """Render the optimized pose graph using matplotlib.

    Parameters
    ----------
    edge_list : list of dict
        Original relative displacement measurements. Each element is a dict with key (i, j) -> (dx, dy).
    positions_df : pd.DataFrame
        DataFrame with columns ['Node', 'X', 'Y'] as returned by `solve_pose_graph`.
    show_measurements : bool, optional
        If True, draw measurement vectors (in red) in addition to optimized edges (in black).
    """

    # Prepare lookup for node coordinates
    coord_lookup = positions_df.set_index('Node')[['X', 'Y']].to_dict('index')

    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot nodes
    xs = positions_df['X'].to_numpy()
    ys = positions_df['Y'].to_numpy()
    ax.scatter(xs, ys, c='tab:blue', zorder=3)

    # Annotate nodes with their indices
    for node, coord in coord_lookup.items():
        ax.annotate(str(node), (coord['X'], coord['Y']), textcoords="offset points", xytext=(0, 5), ha='center')

    # Plot edges and, optionally, measurement arrows
    for entry in edge_list:
        for (i, j), (dx, dy) in entry.items():
            xi, yi = coord_lookup[i]['X'], coord_lookup[i]['Y']
            xj, yj = coord_lookup[j]['X'], coord_lookup[j]['Y']

            # Optimized edge (black line)
            ax.plot([xi, xj], [yi, yj], color='k', linewidth=1.0, alpha=0.6, zorder=1)

            # Measurement arrow (red)
            if show_measurements:
                ax.arrow(xi, yi, dx, dy, color='tab:red', head_width=0.05, length_includes_head=True, alpha=0.7, zorder=2)

    ax.set_aspect('equal', adjustable='datalim')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Pose Graph Optimisation Result')
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()

# ---------------- Synthetic data generation ---------------- #

def generate_synthetic_pose_graph(num_nodes: int = 5, avg_degree: int = 4, radius: float = 5.0, noise_std: float = 1, seed: int = 42) -> List[Dict[Tuple[int, int], Tuple[float, float]]]:
    """Generate a synthetic pose graph with noisy pair-wise displacements.

    The nodes are placed evenly on a circle so that the ground-truth is easy to
    visualise.  Edges are sampled randomly until the expected average degree is
    reached.  Each measurement is the true displacement plus i.i.d. Gaussian
    noise.

    Returns
    -------
    edge_list : list of dict
        In the same format expected by `solve_pose_graph`.
    """
    rng = np.random.default_rng(seed)

    # Ground-truth node positions on a circle
    angles = np.linspace(0, 2 * np.pi, num_nodes, endpoint=False)
    positions = radius * np.stack([np.cos(angles), np.sin(angles)], axis=1)

    # Desired number of undirected edges (E = N * k / 2)
    required_edges = max(1, int(num_nodes * avg_degree / 2))

    chosen_edges = set()
    while len(chosen_edges) < required_edges:
        i = rng.integers(0, num_nodes)
        j = rng.integers(0, num_nodes)
        if i == j:
            continue
        # Treat edges as undirected when checking uniqueness
        if (i, j) in chosen_edges or (j, i) in chosen_edges:
            continue
        chosen_edges.add((i, j))

    edge_list: List[Dict[Tuple[int, int], Tuple[float, float]]] = []
    for i, j in sorted(chosen_edges):
        true_disp = positions[j] - positions[i]
        noisy_disp = true_disp + rng.normal(0.0, noise_std, size=2)
        edge_list.append({(int(i), int(j)): (float(noisy_disp[0]), float(noisy_disp[1]))})

    return edge_list

# ---------------- Example Demo ---------------- #

if __name__ == "__main__":
    # Generate a synthetic dataset with 5 nodes and average degree ~4
    example_input = generate_synthetic_pose_graph(num_nodes=5, avg_degree=4)

    # Compute poses
    df_result = solve_pose_graph(example_input)

    print("Optimised node positions:\n", df_result)

    # Visualise the result
    visualize_pose_graph(example_input, df_result)


