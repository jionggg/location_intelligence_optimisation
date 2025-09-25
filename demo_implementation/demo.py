"""
demo.py

Live 3D PGO Demo with Real-Time UWB Data
=========================================

Integrates all our PGO components:
- MQTT live data ingestion from UWB anchors
- 3D Pose Graph Optimization with anchor-anchor edges
- Real-time position tracking and visualization
- 1Hz sliding window data processing

This is the complete system working with live data!
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import json
import signal
import sys
import threading
from datetime import datetime

# Import MQTT components from uwb_pgo.py
import paho.mqtt.client as mqtt

# Import our PGO pipeline components
from ingest_data import DataIngestor, apply_anchoring_transformation, extract_phone_position
from pgo_3d import solve_pose_graph_3d, create_anchor_anchoring
from create_anchor_edges import ANCHORS, SQUARE_CM, NX, NY

# MQTT Configuration (from uwb_pgo.py)
MQTT_BROKER = "192.168.99.3"
MQTT_PORT = 1883
MQTT_USERNAME = "laptop"
MQTT_PASSWORD = "laptop"
MQTT_CLIENT_ID = "pgo_live_demo"
MQTT_BASE_TOPIC = "uwb"
MQTT_QOS = 0
MQTT_KEEPALIVE = 60

# Room dimensions for plotting
ROOM_W = NX * SQUARE_CM
ROOM_H = NY * SQUARE_CM


class LivePGODemo:
    """
    Live 3D PGO demonstration with real-time MQTT data and visualization.
    """
    
    def __init__(self):
        # PGO components
        self.data_ingestor = DataIngestor(window_size_seconds=1.0)
        
        # Live data tracking
        self.position_history = deque(maxlen=100)  # Keep last 100 positions
        self.timestamps = deque(maxlen=100)
        self.current_position = None
        self.last_update_time = None
        
        # MQTT client
        self.mqtt_client = None
        
        # Visualization
        self.fig = None
        self.ax = None
        self.animation = None
        
        # Thread safety
        self.data_lock = threading.Lock()
        
        # Status tracking
        self.total_measurements = 0
        self.successful_optimizations = 0
        self.last_optimization_time = None
        
    def setup_mqtt(self):
        """Initialize MQTT connection for live UWB data."""
        def on_connect(client, userdata, flags, rc):
            if rc == 0:
                topic = f"{MQTT_BASE_TOPIC}/anchor/+/vector"
                client.subscribe(topic, qos=MQTT_QOS)
                print(f"✅ MQTT connected, subscribed to: {topic}")
            else:
                print(f"❌ MQTT connection failed with code: {rc}")
        
        def on_message(client, userdata, msg):
            try:
                # Parse topic to get anchor ID
                parts = msg.topic.split('/')
                if len(parts) >= 4 and parts[-1] == 'vector' and parts[-3] == 'anchor':
                    anchor_id = int(parts[-2])
                else:
                    return
                
                # Parse message payload
                payload = json.loads(msg.payload.decode('utf-8'))
                recv_time_s = time.time()
                
                # Extract vector data
                vector_data = payload.get('vector_local', {})
                if not all(k in vector_data for k in ['x', 'y', 'z']):
                    return
                
                local_vector = np.array([
                    float(vector_data['x']),
                    float(vector_data['y']),
                    float(vector_data['z'])
                ])
                
                # Add to PGO pipeline
                with self.data_lock:
                    self.data_ingestor.add_measurement(recv_time_s, anchor_id, local_vector)
                    self.total_measurements += 1
                
                print(f"📡 Anchor {anchor_id}: {local_vector} (total: {self.total_measurements})")
                
            except Exception as e:
                print(f"❌ Error processing MQTT message: {e}")
        
        def on_disconnect(client, userdata, rc):
            print("📡 Disconnected from MQTT broker")
        
        # Create and configure MQTT client
        self.mqtt_client = mqtt.Client(
            client_id=MQTT_CLIENT_ID,
            protocol=mqtt.MQTTv311
        )
        
        if MQTT_USERNAME and MQTT_PASSWORD:
            self.mqtt_client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
        
        self.mqtt_client.on_connect = on_connect
        self.mqtt_client.on_message = on_message
        self.mqtt_client.on_disconnect = on_disconnect
        
        # Connect to broker
        try:
            self.mqtt_client.connect(MQTT_BROKER, MQTT_PORT, MQTT_KEEPALIVE)
            self.mqtt_client.loop_start()
            return True
        except Exception as e:
            print(f"❌ Failed to connect to MQTT broker: {e}")
            return False
    
    def run_pgo_optimization(self):
        """Run 3D PGO optimization on current data."""
        try:
            with self.data_lock:
                graph_data = self.data_ingestor.get_latest_graph_data()
            
            if graph_data is None:
                return None
            
            # Set up anchoring
            anchoring = create_anchor_anchoring(ANCHORS[3], ANCHORS[0])
            
            # Run 3D PGO
            optimized_positions = solve_pose_graph_3d(graph_data, anchoring)
            
            # Apply anchoring transformation
            anchored_positions = apply_anchoring_transformation(optimized_positions)
            
            # Extract phone position
            phone_node_id = graph_data['binned_data'].phone_node_id
            x, y, z = extract_phone_position(anchored_positions, phone_node_id)
            
            self.successful_optimizations += 1
            self.last_optimization_time = time.time()
            
            return (x, y, z)
            
        except Exception as e:
            print(f"❌ PGO optimization failed: {e}")
            return None
    
    def update_position(self):
        """Update current position using PGO optimization."""
        new_position = self.run_pgo_optimization()
        
        if new_position is not None:
            with self.data_lock:
                self.current_position = new_position
                self.position_history.append(new_position)
                self.timestamps.append(time.time())
                self.last_update_time = time.time()
            
            x, y, z = new_position
            print(f"🎯 Phone position: ({x:.1f}, {y:.1f}, {z:.1f}) cm")
            
            return True
        return False
    
    def setup_visualization(self):
        """Set up real-time matplotlib visualization."""
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        
        # Draw room grid
        for x in np.arange(0, ROOM_W + 1e-6, SQUARE_CM):
            self.ax.plot([x, x], [0, ROOM_H], linewidth=0.5, color='gray', alpha=0.3)
        for y in np.arange(0, ROOM_H + 1e-6, SQUARE_CM):
            self.ax.plot([0, ROOM_W], [y, y], linewidth=0.5, color='gray', alpha=0.3)
        
        # Draw anchors
        anchor_x = [ANCHORS[i][0] for i in range(4)]
        anchor_y = [ANCHORS[i][1] for i in range(4)]
        self.ax.scatter(anchor_x, anchor_y, s=200, c='red', marker='s', 
                       label="UWB Anchors", zorder=5)
        
        for i in range(4):
            self.ax.text(ANCHORS[i][0], ANCHORS[i][1], f"A{i}",
                        ha="center", va="center", fontsize=12, fontweight='bold')
        
        # Set up plot properties
        self.ax.set_xlim(-50, ROOM_W + 50)
        self.ax.set_ylim(-50, ROOM_H + 50)
        self.ax.set_xlabel("X (cm)", fontsize=12)
        self.ax.set_ylabel("Y (cm)", fontsize=12)
        self.ax.set_title("Live 3D PGO Phone Tracking", fontsize=14, fontweight='bold')
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()
        
        # Initialize empty plots for trajectory and current position
        self.trajectory_plot, = self.ax.plot([], [], 'b-', alpha=0.6, linewidth=2, 
                                           label="Trajectory")
        self.current_plot = self.ax.scatter([], [], s=300, c='yellow', marker='*', 
                                          edgecolors='black', linewidth=2,
                                          label="Current Position", zorder=10)
        
        # Status text
        self.status_text = self.ax.text(0.02, 0.98, "", transform=self.ax.transAxes,
                                       verticalalignment='top', fontsize=10,
                                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
    
    def animate_update(self, frame):
        """Animation update function for matplotlib."""
        with self.data_lock:
            positions = list(self.position_history)
            current_pos = self.current_position
            total_meas = self.total_measurements
            success_opt = self.successful_optimizations
            last_update = self.last_update_time
        
        # Update trajectory
        if len(positions) > 1:
            x_coords = [pos[0] for pos in positions]
            y_coords = [pos[1] for pos in positions]
            self.trajectory_plot.set_data(x_coords, y_coords)
        
        # Update current position
        if current_pos is not None:
            self.current_plot.set_offsets([[current_pos[0], current_pos[1]]])
        else:
            self.current_plot.set_offsets([])
        
        # Update status text
        status_lines = [
            f"📡 Total Measurements: {total_meas}",
            f"🎯 Successful PGO: {success_opt}",
            f"📊 Position History: {len(positions)}",
        ]
        
        if last_update:
            time_since = time.time() - last_update
            status_lines.append(f"⏱️  Last Update: {time_since:.1f}s ago")
        
        if current_pos:
            x, y, z = current_pos
            status_lines.append(f"📍 Position: ({x:.1f}, {y:.1f}, {z:.1f}) cm")
        
        self.status_text.set_text('\n'.join(status_lines))
        
        return [self.trajectory_plot, self.current_plot, self.status_text]
    
    def start_live_demo(self):
        """Start the complete live demo system."""
        print("🚀 Starting Live 3D PGO Demo...")
        print("=" * 50)
        
        # Setup MQTT connection
        if not self.setup_mqtt():
            print("❌ Failed to setup MQTT connection. Exiting.")
            return
        
        # Setup visualization
        self.setup_visualization()
        
        # Start PGO processing thread
        def pgo_worker():
            while True:
                try:
                    self.update_position()
                    time.sleep(1.0)  # 1Hz updates
                except Exception as e:
                    print(f"❌ PGO worker error: {e}")
                    time.sleep(1.0)
        
        pgo_thread = threading.Thread(target=pgo_worker, daemon=True)
        pgo_thread.start()
        
        # Start animation
        self.animation = animation.FuncAnimation(
            self.fig, self.animate_update, interval=500, blit=False
        )
        
        print("📊 Visualization started. Close the plot window to exit.")
        print("📡 Waiting for UWB data from MQTT...")
        print("🎯 PGO optimization will start automatically when data arrives.")
        print("=" * 50)
        
        # Show plot (blocking)
        plt.show()
    
    def cleanup(self):
        """Clean up resources."""
        if self.mqtt_client:
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()
        
        if self.animation:
            self.animation.event_source.stop()
        
        print("\n🛑 Demo stopped. Resources cleaned up.")


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    print("\n🛑 Received interrupt signal. Cleaning up...")
    sys.exit(0)


def main():
    """Main demo function."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                  🎯 Live 3D PGO Demo                        ║
║                                                              ║
║  Real-time UWB positioning with 3D Pose Graph Optimization  ║
║                                                              ║
║  Features:                                                   ║
║  • Live MQTT data ingestion from UWB anchors               ║
║  • 3D PGO with anchor-anchor constraints                   ║
║  • 1Hz sliding window processing                           ║
║  • Real-time position tracking & visualization             ║
║                                                              ║
║  Press Ctrl+C to exit                                       ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Create and start demo
    demo = LivePGODemo()
    
    try:
        demo.start_live_demo()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"❌ Demo error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        demo.cleanup()


if __name__ == "__main__":
    main()
