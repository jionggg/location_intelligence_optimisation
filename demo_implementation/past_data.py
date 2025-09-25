"""
past_data.py

Test script that processes real UWB data from uwb_data_A_20250916_175743.csv
Uses the first 20 rows and runs the complete PGO pipeline.
"""

import pandas as pd
import numpy as np
import time
import os
import matplotlib.pyplot as plt

# Import our PGO pipeline components
from ingest_data import DataIngestor, apply_anchoring_transformation, extract_phone_position
from pgo_3d import solve_pose_graph_3d, create_anchor_anchoring
from create_anchor_edges import ANCHORS, SQUARE_CM, NX, NY

# Room dimensions for plotting
ROOM_W = NX * SQUARE_CM
ROOM_H = NY * SQUARE_CM


def load_csv_data(csv_path: str, num_rows: int = 20) -> pd.DataFrame:
    """Load the first N rows from the UWB CSV file."""
    df = pd.read_csv(csv_path, nrows=num_rows)
    print(f"Loaded {len(df)} rows from {csv_path}")
    return df


def parse_measurements_from_csv(df: pd.DataFrame) -> list:
    """
    Parse UWB measurements from CSV dataframe.

    Returns list of (timestamp_seconds, anchor_id, local_vector) tuples.
    """
    measurements = []

    # For testing purposes, group all measurements at the same timestamp
    # so they get binned together in the sliding window
    test_timestamp = time.time()  # Use current time as reference

    for idx, row in df.iterrows():
        # Parse measurements for each anchor
        for anchor_id in range(4):
            local_x_col = f'anchor_{anchor_id}_local_x'
            local_y_col = f'anchor_{anchor_id}_local_y'
            local_z_col = f'anchor_{anchor_id}_local_z'

            # Check if we have valid measurements for this anchor
            if pd.notna(row[local_x_col]) and pd.notna(row[local_y_col]) and pd.notna(row[local_z_col]):
                local_vector = np.array([
                    float(row[local_x_col]),
                    float(row[local_y_col]),
                    float(row[local_z_col])
                ])

                # Use the same timestamp for all measurements so they get binned together
                measurements.append((test_timestamp, anchor_id, local_vector))

    print(f"Parsed {len(measurements)} total measurements from {len(df)} rows")
    return measurements


def run_pgo_pipeline(measurements: list, verbose: bool = True) -> dict:
    """
    Run the complete PGO pipeline on the measurements.

    Returns dict with optimized positions and metadata.
    """
    if verbose:
        print("\n=== Starting PGO Pipeline ===")

        # Debug: Show sample measurements
        print("Sample measurements:")
        for i, (timestamp, anchor_id, local_vec) in enumerate(measurements[:4]):
            print(f"  {i}: anchor_{anchor_id} -> {local_vec}")

    # 1. Initialize data ingestor
    ingestor = DataIngestor(window_size_seconds=1.0)

    # 2. Feed measurements to ingestor
    if verbose:
        print("Feeding measurements to ingestor...")
    for timestamp, anchor_id, local_vec in measurements:
        ingestor.add_measurement(timestamp, anchor_id, local_vec)

    # 3. Get graph data
    graph_data = ingestor.get_latest_graph_data()
    if graph_data is None:
        raise ValueError("No graph data generated - check measurements")

    if verbose:
        print(f"Generated graph with {len(graph_data['edges'])} edges")
        
        # Debug: Show anchor-phone edges
        print("Anchor-phone edges:")
        for edge in graph_data['edges']:
            if edge[0].startswith('anchor_') and edge[1].startswith('phone_'):
                print(f"  {edge[0]} -> {edge[1]}: {edge[2]}")

    # 4. Set up anchoring (anchor_3 at origin)
    anchoring = create_anchor_anchoring(ANCHORS[3], ANCHORS[0])

    # 5. Run 3D PGO
    if verbose:
        print("Running 3D PGO optimization...")
    optimized_positions = solve_pose_graph_3d(graph_data, anchoring)
    
    if verbose:
        # Debug: Show optimized positions before anchoring
        print("Optimized positions (before anchoring):")
        for node_id, pos in optimized_positions.items():
            print(f"  {node_id}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")

    # 6. Apply anchoring transformation
    if verbose:
        print("Applying anchoring transformation...")
    anchored_positions = apply_anchoring_transformation(optimized_positions)
    
    if verbose:
        # Debug: Show positions after anchoring
        print("Anchored positions (after transformation):")
        for node_id, pos in anchored_positions.items():
            print(f"  {node_id}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")

    # 7. Extract final phone position
    phone_node_id = graph_data['binned_data'].phone_node_id
    final_x, final_y, final_z = extract_phone_position(anchored_positions, phone_node_id)

    return {
        'graph_data': graph_data,
        'optimized_positions': optimized_positions,
        'anchored_positions': anchored_positions,
        'final_position': (final_x, final_y, final_z),
        'phone_node_id': phone_node_id
    }


def plot_results(results: dict, output_dir: str = "test_output"):
    """
    Plot the PGO results similar to uwb_pgo.py
    """
    os.makedirs(output_dir, exist_ok=True)

    # Extract data
    anchored_positions = results['anchored_positions']
    phone_node_id = results['phone_node_id']

    # Get phone trajectory (just one point in this case since we're using 1-second binning)
    if phone_node_id in anchored_positions:
        phone_pos = anchored_positions[phone_node_id]
        phone_positions = np.array([phone_pos])  # Single point
    else:
        print("Warning: Phone position not found in results")
        phone_positions = np.array([])

    # Create plot similar to uwb_pgo.py
    fig, ax = plt.subplots(figsize=(7, 6))

    # Draw grid
    for x in np.arange(0, ROOM_W + 1e-6, SQUARE_CM):
        ax.plot([x, x], [0, ROOM_H], linewidth=0.5, color='gray', alpha=0.5)
    for y in np.arange(0, ROOM_H + 1e-6, SQUARE_CM):
        ax.plot([0, ROOM_W], [y, y], linewidth=0.5, color='gray', alpha=0.5)

    # Plot anchors
    ax.scatter([ANCHORS[i][0] for i in range(4)],
               [ANCHORS[i][1] for i in range(4)],
               s=100, c='red', marker='s', label="Anchors", zorder=3)
    for idx in range(4):
        ax.text(ANCHORS[idx][0], ANCHORS[idx][1], f"A{idx}",
                ha="center", va="center", fontsize=12, fontweight='bold')

    # Plot optimized anchor positions (after anchoring)
    optimized_anchors = []
    for i in range(4):
        anchor_id = f"anchor_{i}"
        if anchor_id in anchored_positions:
            pos = anchored_positions[anchor_id]
            optimized_anchors.append(pos)
            ax.scatter([pos[0]], [pos[1]], s=80, c='blue', marker='^',
                      label="Optimized Anchors" if i == 0 else "", zorder=2)

    # Plot phone position
    if len(phone_positions) > 0:
        ax.scatter(phone_positions[:, 0], phone_positions[:, 1],
                  s=50, c='green', marker='o', label="Phone Position", zorder=4)

        # Mark the final position prominently
        final_pos = phone_positions[0]
        ax.scatter([final_pos[0]], [final_pos[1]], s=200, c='yellow',
                  marker='*', edgecolors='black', linewidth=2,
                  label="Final Position", zorder=5)

        ax.text(final_pos[0], final_pos[1],
                f"({final_pos[0]:.1f}, {final_pos[1]:.1f})",
                ha="left", va="bottom", fontsize=10)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-50, ROOM_W + 50)
    ax.set_ylim(-50, ROOM_H + 50)
    ax.set_xlabel("X (cm)")
    ax.set_ylabel("Y (cm)")
    ax.set_title("3D PGO Results - Real UWB Data Test")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Save plot
    output_path = os.path.join(output_dir, "pgo_real_data_test.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved plot to: {output_path}")
    plt.close(fig)

    return output_path


def test_multiple_datasets():
    """Test with different parts of the dataset."""
    csv_path = "/Users/hongyilin/projects/location_intelligence_optimisation/wrecess-tue/sept16uwbdata/uwb_data_A_20250916_175743.csv"
    
    # Test different ranges of the dataset
    test_ranges = [
        (0, 20, "First 20 rows"),
        (50, 70, "Rows 50-70"),
        (100, 120, "Rows 100-120"),
        (200, 220, "Rows 200-220"),
        (400, 420, "Rows 400-420"),
    ]
    
    results_summary = []
    
    for start_row, end_row, description in test_ranges:
        print(f"\n{'='*60}")
        print(f"Testing {description} (rows {start_row}-{end_row})")
        print('='*60)
        
        try:
            # Load specific range of data
            df_full = pd.read_csv(csv_path)
            df = df_full.iloc[start_row:end_row].copy()
            print(f"Loaded {len(df)} rows from {csv_path}")
            
            # Parse measurements
            measurements = parse_measurements_from_csv(df)
            
            if not measurements:
                print("No valid measurements found in this range")
                continue
                
            # Run PGO pipeline (quiet mode)
            results = run_pgo_pipeline(measurements, verbose=False)
            
            # Store results
            final_x, final_y, final_z = results['final_position']
            results_summary.append({
                'range': description,
                'position': (final_x, final_y, final_z),
                'measurements_count': len(measurements)
            })
            
            print(f"Final position: ({final_x:.1f}, {final_y:.1f}, {final_z:.1f}) cm")
            
        except Exception as e:
            print(f"Error processing {description}: {e}")
            results_summary.append({
                'range': description,
                'position': 'ERROR',
                'measurements_count': 0
            })
    
    # Summary of all results
    print(f"\n{'='*60}")
    print("SUMMARY OF ALL TESTS")
    print('='*60)
    print(f"{'Range':<20} {'Position (x,y,z) cm':<25} {'Measurements'}")
    print('-'*60)
    
    for result in results_summary:
        if result['position'] != 'ERROR':
            x, y, z = result['position']
            pos_str = f"({x:.1f}, {y:.1f}, {z:.1f})"
        else:
            pos_str = "ERROR"
        print(f"{result['range']:<20} {pos_str:<25} {result['measurements_count']}")
    
    return results_summary


def main():
    """Main test function."""
    print("=== Real UWB Data PGO Test - Multiple Datasets ===")

    # Path to CSV file
    csv_path = "/Users/hongyilin/projects/location_intelligence_optimisation/wrecess-tue/sept16uwbdata/uwb_data_A_20250916_175743.csv"

    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return

    # Test multiple ranges
    results_summary = test_multiple_datasets()
    
    # Generate plot for the first successful result
    print(f"\n{'='*60}")
    print("Generating plot for first successful result...")
    
    try:
        # Use first 20 rows for plotting
        df = load_csv_data(csv_path, num_rows=20)
        measurements = parse_measurements_from_csv(df)
        
        if measurements:
            results = run_pgo_pipeline(measurements)
            plot_path = plot_results(results, output_dir="test_output")
            print(f"Plot saved to: {plot_path}")
        
    except Exception as e:
        print(f"Error generating plot: {e}")
    
    print("\n✅ Multiple dataset test completed!")


if __name__ == "__main__":
    main()
