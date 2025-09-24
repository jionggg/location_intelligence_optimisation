"""
ingest_data.py

Data ingestion and graph construction for real-time PGO.
Handles time series data from MQTT, transforms to relative measurements,
constructs pose graph with anchor-anchor edges, and bins data using sliding window.
"""

import time
import numpy as np
from collections import defaultdict, deque
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

# Import our transformation functions
from transform_to_global_vector import create_relative_measurement
from create_anchor_edges import create_anchor_anchor_edges, ANCHORS


@dataclass
class Measurement:
    """Single UWB measurement from one anchor at one timestamp."""
    timestamp: float
    anchor_id: int
    local_vector: np.ndarray  # [x, y, z] in cm


@dataclass
class BinnedData:
    """Data for one 1-second bin."""
    bin_start_time: float
    bin_end_time: float
    measurements: Dict[int, List[np.ndarray]]  # anchor_id -> list of vectors
    phone_node_id: str

    def get_averaged_measurements(self) -> List[Tuple[str, str, np.ndarray]]:
        """Get averaged measurements for each anchor in this bin."""
        edges = []
        for anchor_id, vectors in self.measurements.items():
            if vectors:  # Only if we have measurements for this anchor
                avg_vector = np.mean(vectors, axis=0)  # Average all vectors for this anchor
                edge = create_relative_measurement(anchor_id, self.phone_node_id, avg_vector)
                edges.append(edge)
        return edges


class DataIngestor:
    """
    Handles data ingestion, binning, and graph construction for real-time PGO.

    Maintains a sliding window of measurements and creates binned data ready for PGO.
    """

    def __init__(self, window_size_seconds: float = 1.0):
        """
        Args:
            window_size_seconds: Size of sliding window in seconds (default 1.0 for 1Hz updates)
        """
        self.window_size_seconds = window_size_seconds
        self.measurements_buffer = deque()  # Sliding window of raw measurements
        self.bin_counter = 0  # For generating unique phone node IDs

        # Always include anchor-anchor edges (perfect constraints)
        self.anchor_anchor_edges = create_anchor_anchor_edges()

    def add_measurement(self, timestamp: float, anchor_id: int, local_vector: np.ndarray):
        """
        Add a new UWB measurement to the buffer.

        Args:
            timestamp: Measurement timestamp (seconds)
            anchor_id: Anchor ID (0-3)
            local_vector: [x, y, z] vector in anchor's local coordinates (cm)
        """
        measurement = Measurement(timestamp, anchor_id, local_vector)
        self.measurements_buffer.append(measurement)

        # Remove old measurements outside the sliding window
        current_time = time.time()
        while self.measurements_buffer and \
              (current_time - self.measurements_buffer[0].timestamp) > self.window_size_seconds:
            self.measurements_buffer.popleft()

    def create_binned_data(self) -> Optional[BinnedData]:
        """
        Create binned data from current measurements in the sliding window.

        Returns:
            BinnedData if there are measurements, None otherwise
        """
        if not self.measurements_buffer:
            return None

        # Get time range of current window
        current_time = time.time()
        window_start = current_time - self.window_size_seconds

        # Group measurements by anchor within the current window
        anchor_measurements = defaultdict(list)

        for measurement in self.measurements_buffer:
            if measurement.timestamp >= window_start:
                anchor_measurements[measurement.anchor_id].append(measurement.local_vector)

        if not anchor_measurements:
            return None

        # Create binned data
        self.bin_counter += 1
        phone_node_id = f"phone_bin_{self.bin_counter}"

        binned_data = BinnedData(
            bin_start_time=window_start,
            bin_end_time=current_time,
            measurements=dict(anchor_measurements),
            phone_node_id=phone_node_id
        )

        return binned_data

    def create_graph_data(self, binned_data: BinnedData) -> Dict:
        """
        Create complete graph data ready for PGO from binned measurements.

        Args:
            binned_data: Binned measurement data

        Returns:
            Dict with 'nodes' and 'edges' ready for PGO
        """
        # Nodes: anchors (floating) + phone pose (to optimize)
        # Anchors start floating - their positions will be determined by anchor-anchor edges
        nodes = {
            'anchor_0': None,  # floating - constrained by edges
            'anchor_1': None,  # floating - constrained by edges
            'anchor_2': None,  # floating - constrained by edges
            'anchor_3': None,  # floating - will be pinned in PGO solver
            binned_data.phone_node_id: None  # unknown, to be optimized
        }

        # Edges: anchor-phone + anchor-anchor
        edges = []

        # Add anchor-phone edges from binned measurements
        anchor_phone_edges = binned_data.get_averaged_measurements()
        edges.extend(anchor_phone_edges)

        # Add anchor-anchor edges (perfect constraints between anchors)
        edges.extend(self.anchor_anchor_edges)

        return {
            'nodes': nodes,
            'edges': edges,
            'binned_data': binned_data
        }

    def get_latest_graph_data(self) -> Optional[Dict]:
        """
        Get the latest graph data ready for PGO.

        Returns:
            Dict with nodes, edges, and binned_data if measurements available, None otherwise
        """
        binned_data = self.create_binned_data()
        if binned_data is None:
            return None

        return self.create_graph_data(binned_data)


def apply_anchoring_transformation(optimized_nodes: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Apply anchoring transformation to align optimized positions with known anchor positions.
    Since we now anchor both anchor_3 and anchor_2 during PGO, this function mainly
    ensures all anchors are at their exact known positions.

    Args:
        optimized_nodes: Dict of optimized node positions from PGO

    Returns:
        Dict of anchored positions where all anchors are fixed to their true positions
    """
    # Extract optimized anchor positions
    opt_anchor_3 = optimized_nodes.get('anchor_3')
    opt_anchor_0 = optimized_nodes.get('anchor_0')

    if opt_anchor_3 is None or opt_anchor_0 is None:
        raise ValueError("Optimized anchor positions not found")

    # Target positions
    target_anchor_3 = ANCHORS[3]  # (0,0,0)
    target_anchor_0 = ANCHORS[0]  # (440,550,0)

    # Step 1: Translate so anchor_3 is at origin
    translation = -opt_anchor_3
    translated_nodes = {}
    for node_id, position in optimized_nodes.items():
        if position is not None:
            translated_nodes[node_id] = position + translation

    # After translation, anchor_3 should be at origin
    translated_anchor_0 = translated_nodes['anchor_0']
    
    # Step 2: Calculate scale factor
    # Distance between optimized anchors 0 and 3
    opt_distance = np.linalg.norm(translated_anchor_0)
    # True distance between anchors 0 and 3
    true_distance = np.linalg.norm(target_anchor_0 - target_anchor_3)
    
    if opt_distance > 1e-6:  # Avoid division by zero
        scale_factor = true_distance / opt_distance
    else:
        scale_factor = 1.0
    
    # Step 3: Calculate rotation
    # We need to rotate translated_anchor_0 to align with target_anchor_0
    if opt_distance > 1e-6:
        # Normalized vectors
        opt_direction = translated_anchor_0 / opt_distance
        target_direction = target_anchor_0 / true_distance
        
        # Calculate rotation matrix using Rodrigues' rotation formula
        # For 2D case (Z components are 0), we can use a simple 2D rotation
        if abs(opt_direction[2]) < 1e-6 and abs(target_direction[2]) < 1e-6:
            # 2D rotation in XY plane
            opt_angle = np.arctan2(opt_direction[1], opt_direction[0])
            target_angle = np.arctan2(target_direction[1], target_direction[0])
            rotation_angle = target_angle - opt_angle
            
            cos_theta = np.cos(rotation_angle)
            sin_theta = np.sin(rotation_angle)
            
            rotation_matrix = np.array([
                [cos_theta, -sin_theta, 0],
                [sin_theta,  cos_theta, 0],
                [0,          0,         1]
            ])
        else:
            # 3D rotation using cross product
            v = np.cross(opt_direction, target_direction)
            s = np.linalg.norm(v)
            c = np.dot(opt_direction, target_direction)
            
            if s < 1e-6:  # Vectors are parallel
                if c > 0:
                    rotation_matrix = np.eye(3)
                else:
                    rotation_matrix = -np.eye(3)
            else:
                vx = np.array([[0, -v[2], v[1]],
                              [v[2], 0, -v[0]],
                              [-v[1], v[0], 0]])
                rotation_matrix = np.eye(3) + vx + np.dot(vx, vx) * ((1 - c) / (s * s))
    else:
        rotation_matrix = np.eye(3)

    # Step 4: Apply scale and rotation to all nodes
    transformed_nodes = {}
    for node_id, position in translated_nodes.items():
        if position is not None:
            # Apply scaling
            scaled_pos = position * scale_factor
            # Apply rotation
            rotated_pos = rotation_matrix @ scaled_pos
            transformed_nodes[node_id] = rotated_pos

    # Step 5: Force known anchors to their exact positions (remove numerical errors)
    transformed_nodes['anchor_3'] = ANCHORS[3]  # (0,0,0)
    transformed_nodes['anchor_0'] = ANCHORS[0]  # (440,550,0)
    
    # Also fix other anchors to their known positions
    transformed_nodes['anchor_1'] = ANCHORS[1]  # (0,550,0)
    transformed_nodes['anchor_2'] = ANCHORS[2]  # (440,0,0)

    return transformed_nodes


def extract_phone_position(anchored_nodes: Dict[str, np.ndarray], phone_node_id: str) -> Tuple[float, float, float]:
    """
    Extract the final phone position from anchored nodes.

    Args:
        anchored_nodes: Dict of anchored node positions
        phone_node_id: ID of the phone node to extract

    Returns:
        Tuple of (x, y, z) position in cm
    """
    position = anchored_nodes.get(phone_node_id)
    if position is None:
        raise ValueError(f"Phone position not found for node {phone_node_id}")

    return (float(position[0]), float(position[1]), float(position[2]))


# Example usage and testing
if __name__ == "__main__":
    print("Testing DataIngestor...")

    # Create ingestor
    ingestor = DataIngestor(window_size_seconds=1.0)

    # Simulate some measurements over time
    current_time = time.time()

    # Add some test measurements
    test_measurements = [
        (current_time + 0.1, 0, np.array([10.0, 20.0, 5.0])),
        (current_time + 0.2, 1, np.array([15.0, 25.0, 6.0])),
        (current_time + 0.3, 2, np.array([12.0, 22.0, 4.0])),
        (current_time + 0.4, 3, np.array([8.0, 18.0, 7.0])),
        (current_time + 0.6, 0, np.array([11.0, 21.0, 5.5])),  # Second measurement from anchor 0
        (current_time + 0.8, 1, np.array([16.0, 26.0, 6.5])),  # Second measurement from anchor 1
    ]

    for timestamp, anchor_id, local_vec in test_measurements:
        ingestor.add_measurement(timestamp, anchor_id, local_vec)

    # Get graph data
    graph_data = ingestor.get_latest_graph_data()

    if graph_data:
        print(f"\nNodes: {list(graph_data['nodes'].keys())}")
        print("Node positions (None = floating, to be optimized):")
        for node_id, position in graph_data['nodes'].items():
            status = "FLOATING" if position is None else f"FIXED at {position}"
            print(f"  {node_id}: {status}")

        print(f"\nNumber of edges: {len(graph_data['edges'])}")

        print("\nAnchor-phone edges:")
        for edge in graph_data['edges']:
            if edge[0].startswith('anchor_') and edge[1].startswith('phone_'):
                print(f"  {edge[0]} -> {edge[1]}: {edge[2]}")

        print("\nAnchor-anchor edges:")
        anchor_edges_count = 0
        for edge in graph_data['edges']:
            if edge[0].startswith('anchor_') and edge[1].startswith('anchor_'):
                anchor_edges_count += 1
        print(f"  {anchor_edges_count} anchor-anchor edges included")

        print(f"\nBinned data contains measurements from {len(graph_data['binned_data'].measurements)} anchors")

        # Test anchoring functions
        print("\n--- Testing Anchoring Functions ---")

        # Simulate PGO results (normally from PGO solver)
        simulated_pgo_results = {
            'anchor_0': np.array([100.0, 200.0, 0.0]),  # Simulated optimized position
            'anchor_1': np.array([50.0, 200.0, 0.0]),
            'anchor_2': np.array([100.0, 50.0, 0.0]),
            'anchor_3': np.array([50.0, 50.0, 0.0]),    # Simulated optimized position
            'phone_bin_1': np.array([75.0, 125.0, 10.0]) # Simulated phone position
        }

        print("Simulated PGO results:")
        for node_id, pos in simulated_pgo_results.items():
            print(f"  {node_id}: {pos}")

        # Apply anchoring transformation
        anchored_results = apply_anchoring_transformation(simulated_pgo_results)
        print("\nAfter anchoring transformation:")
        for node_id, pos in anchored_results.items():
            print(f"  {node_id}: {pos}")

        # Extract phone position
        phone_x, phone_y, phone_z = extract_phone_position(anchored_results, 'phone_bin_1')
        print(f"\nFinal phone position: ({phone_x:.1f}, {phone_y:.1f}, {phone_z:.1f}) cm")

    else:
        print("No graph data available")
