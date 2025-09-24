"""
ingest_data.py

Data ingestion and sliding window processing for real-time 3D PGO.

Handles:
- Sliding window data collection (1-second window)
- Measurement binning and averaging
- Graph construction with nodes and edges
- Post-PGO anchoring transformations
"""

import time
import numpy as np
from dataclasses import dataclass
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

from transform_to_global_vector import create_relative_measurement
from create_anchor_edges import create_anchor_anchor_edges, ANCHORS


@dataclass
class Measurement:
    """Single UWB measurement."""
    timestamp: float
    anchor_id: int
    local_vector: np.ndarray


@dataclass 
class BinnedData:
    """Binned and averaged measurements for a time window."""
    phone_node_id: str
    anchor_measurements: Dict[int, np.ndarray]  # anchor_id -> averaged_relative_vector
    
    def get_averaged_measurements(self) -> List[Tuple[str, str, np.ndarray]]:
        """Convert to list of relative measurement tuples."""
        edges = []
        for anchor_id, avg_vector in self.anchor_measurements.items():
            anchor_node = f"anchor_{anchor_id}"
            edges.append((anchor_node, self.phone_node_id, avg_vector))
        return edges


class DataIngestor:
    """
    Handles data ingestion, binning, and graph construction for real-time PGO.

    Maintains a sliding window of measurements and creates binned data ready for PGO.
    """
    
    def __init__(self, window_size_seconds: float = 1.0):
        self.window_size_seconds = window_size_seconds
        self.measurements = deque()  # Sliding window of measurements
        self.bin_counter = 0
        
        # Pre-generate anchor-anchor edges (perfect constraints)
        self.anchor_anchor_edges = create_anchor_anchor_edges()
        print(f"Generated {len(self.anchor_anchor_edges)} anchor-anchor edges")
    
    def add_measurement(self, timestamp: float, anchor_id: int, local_vector: np.ndarray):
        """Add a new UWB measurement and maintain sliding window."""
        measurement = Measurement(timestamp, anchor_id, local_vector)
        self.measurements.append(measurement)
        
        # Remove old measurements outside the window
        current_time = time.time()
        window_start = current_time - self.window_size_seconds
        
        while self.measurements and self.measurements[0].timestamp < window_start:
            self.measurements.popleft()
    
    def create_binned_data(self) -> Optional[BinnedData]:
        """
        Create binned data from measurements in the current window.
        
        Returns:
            BinnedData if measurements available, None otherwise
        """
        if not self.measurements:
            return None
        
        # Group measurements by anchor
        anchor_measurements = defaultdict(list)
        for measurement in self.measurements:
            # Convert to relative measurement
            anchor_node, phone_node_id, relative_vector = create_relative_measurement(
                measurement.anchor_id, 
                f"phone_bin_{self.bin_counter}",
                measurement.local_vector
            )
            anchor_measurements[measurement.anchor_id].append(relative_vector)
        
        # Average measurements for each anchor
        averaged_measurements = {}
        for anchor_id, vectors in anchor_measurements.items():
            if vectors:  # Only include anchors with measurements
                averaged_measurements[anchor_id] = np.mean(vectors, axis=0)
        
        if not averaged_measurements:
            return None
        
        phone_node_id = f"phone_bin_{self.bin_counter}"
        self.bin_counter += 1
        
        binned_data = BinnedData(
            anchor_measurements=averaged_measurements,
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
    # Since we now properly anchor anchor_3 and anchor_2 during PGO optimization,
    # the solution should already be in the correct global frame.
    # We just need to ensure all anchors are at their exact positions.
    
    anchored_nodes = optimized_nodes.copy()
    
    # Ensure all anchors are exactly at their known positions
    anchored_nodes['anchor_0'] = ANCHORS[0]  # (440, 550, 0)
    anchored_nodes['anchor_1'] = ANCHORS[1]  # (0, 550, 0)
    anchored_nodes['anchor_2'] = ANCHORS[2]  # (440, 0, 0)
    anchored_nodes['anchor_3'] = ANCHORS[3]  # (0, 0, 0)
    
    return anchored_nodes


def extract_phone_position(anchored_nodes: Dict[str, np.ndarray], phone_node_id: str) -> Tuple[float, float, float]:
    """
    Extract the phone position from anchored node positions.

    Args:
        anchored_nodes: Dict of anchored node positions
        phone_node_id: ID of the phone node to extract

    Returns:
        Tuple of (x, y, z) coordinates in cm
    """
    phone_pos = anchored_nodes.get(phone_node_id)
    if phone_pos is None:
        raise ValueError(f"Phone position not found for node {phone_node_id}")
    
    return float(phone_pos[0]), float(phone_pos[1]), float(phone_pos[2])


# Example usage and testing
if __name__ == "__main__":
    import time
    
    print("=== DataIngestor Test ===")
    
    # Create ingestor
    ingestor = DataIngestor(window_size_seconds=1.0)
    
    # Simulate measurements
    current_time = time.time()
    test_measurements = [
        (current_time - 0.9, 0, np.array([100.0, -50.0, -30.0])),
        (current_time - 0.8, 1, np.array([80.0, 60.0, -25.0])),
        (current_time - 0.7, 2, np.array([120.0, -40.0, -35.0])),
        (current_time - 0.6, 3, np.array([70.0, 20.0, -20.0])),
    ]
    
    # Add measurements
    for timestamp, anchor_id, local_vec in test_measurements:
        ingestor.add_measurement(timestamp, anchor_id, local_vec)
    
    # Get graph data
    graph_data = ingestor.get_latest_graph_data()
    
    if graph_data:
        print(f"✅ Graph created with {len(graph_data['nodes'])} nodes and {len(graph_data['edges'])} edges")
        
        print("Nodes:")
        for node_id, pos in graph_data['nodes'].items():
            print(f"  {node_id}: {pos}")
        
        print("Edges (first 5):")
        for i, (from_node, to_node, vector) in enumerate(graph_data['edges'][:5]):
            print(f"  {from_node} -> {to_node}: {vector}")
        
        # Test anchoring transformation with dummy optimized positions
        dummy_optimized = {
            'anchor_0': np.array([400.0, 500.0, 10.0]),
            'anchor_1': np.array([50.0, 520.0, 5.0]),
            'anchor_2': np.array([420.0, 30.0, 8.0]),
            'anchor_3': np.array([10.0, 20.0, 2.0]),
            'phone_bin_0': np.array([200.0, 250.0, -15.0])
        }
        
        print("\n=== Testing Anchoring Transformation ===")
        anchored = apply_anchoring_transformation(dummy_optimized)
        print("Anchored positions:")
        for node_id, pos in anchored.items():
            print(f"  {node_id}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
        
        # Test phone position extraction
        x, y, z = extract_phone_position(anchored, 'phone_bin_0')
        print(f"\nPhone position: ({x:.3f}, {y:.3f}, {z:.3f}) cm")
        
    else:
        print("❌ No graph data generated")
