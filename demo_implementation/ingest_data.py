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
        # Nodes: anchors (fixed) + phone pose (to optimize)
        nodes = {
            'anchor_0': ANCHORS[0],  # fixed positions
            'anchor_1': ANCHORS[1],
            'anchor_2': ANCHORS[2],
            'anchor_3': ANCHORS[3],
            binned_data.phone_node_id: None  # unknown, to be optimized
        }

        # Edges: anchor-phone + anchor-anchor
        edges = []

        # Add anchor-phone edges from binned measurements
        anchor_phone_edges = binned_data.get_averaged_measurements()
        edges.extend(anchor_phone_edges)

        # Add anchor-anchor edges (perfect constraints)
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
        print(f"Number of edges: {len(graph_data['edges'])}")

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
    else:
        print("No graph data available")
