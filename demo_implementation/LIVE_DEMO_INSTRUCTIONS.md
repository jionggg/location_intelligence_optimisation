# 🎯 Live 3D PGO Demo Instructions

## Prerequisites

✅ **Hardware Setup:**
- UWB anchors positioned in room corners
- UWB tag/phone for tracking
- Laptop with MQTT broker running (`192.168.99.3`)

✅ **Software Requirements:**
- Conda environment `MLenv` with scipy, matplotlib, paho-mqtt
- All PGO components built and tested

## 🚀 Quick Start

### 1. Navigate to Demo Directory
```bash
cd /Users/hongyilin/projects/location_intelligence_optimisation/demo_implementation
```

### 2. Activate Environment
```bash
conda activate MLenv
```

### 3. Run Live Demo
```bash
python demo.py
```

## 🎪 What to Expect

### **Console Output:**
```
╔══════════════════════════════════════════════════════════════╗
║                  🎯 Live 3D PGO Demo                        ║
║  Real-time UWB positioning with 3D Pose Graph Optimization  ║
╚══════════════════════════════════════════════════════════════╝

🚀 Starting Live 3D PGO Demo...
✅ MQTT connected, subscribed to: uwb/anchor/+/vector
📊 Visualization started. Close the plot window to exit.
📡 Waiting for UWB data from MQTT...
🎯 PGO optimization will start automatically when data arrives.
```

### **Live Data Flow:**
```
📡 Anchor 0: [681.64, 167.68, -38.24] (total: 1)
📡 Anchor 1: [430.24, -410.57, -39.19] (total: 2)
📡 Anchor 3: [63.10, -45.85, -135.10] (total: 3)
🎯 Phone position: (370.9, 394.9, 50.1) cm
```

### **Visualization Window:**
- **Room grid** with 55cm squares
- **Red anchor positions** (A0, A1, A2, A3)
- **Yellow star** showing current phone position
- **Blue trajectory line** showing phone movement
- **Status box** with live statistics

## 🔧 System Configuration

### **MQTT Settings** (in `demo.py`):
```python
MQTT_BROKER = "192.168.99.3"      # Your laptop IP
MQTT_PORT = 1883
MQTT_USERNAME = "laptop"
MQTT_PASSWORD = "laptop"
MQTT_BASE_TOPIC = "uwb"
```

### **Expected MQTT Topics:**
- `uwb/anchor/0/vector`
- `uwb/anchor/1/vector`
- `uwb/anchor/2/vector`
- `uwb/anchor/3/vector`

### **Expected JSON Message Format:**
```json
{
  "t_unix_ns": 1758015496360497171,
  "vector_local": {
    "x": 681.6383,
    "y": 167.6787,
    "z": -38.2379
  }
}
```

## 🎯 Room Layout

```
     0 -------- 440cm -------- X
     |                        |
     |  A1 (0,550)    A0 (440,550)  550cm
     |                        |
     |                        |
     |  A3 (0,0)      A2 (440,0)    |
     |                        |
     Y ------------------------

Origin: A3 (bottom-left)
```

## 📊 Live Status Display

The visualization shows:
- **📡 Total Measurements:** Count of UWB vectors received
- **🎯 Successful PGO:** Number of successful optimizations
- **📊 Position History:** Length of trajectory
- **⏱️ Last Update:** Time since last position update
- **📍 Position:** Current (x,y,z) coordinates in cm

## 🛠️ Troubleshooting

### **No MQTT Connection:**
```bash
❌ Failed to connect to MQTT broker: [Errno 61] Connection refused
```
**Solution:** 
- Check laptop IP address: `ifconfig` or `ip addr`
- Update `MQTT_BROKER` in `demo.py` if needed
- Ensure MQTT broker is running on laptop

### **No UWB Data:**
```bash
📡 Waiting for UWB data from MQTT...
```
**Solution:**
- Check UWB anchors are powered and connected
- Verify MQTT topic names match expected format
- Use MQTT client to debug: `mosquitto_sub -h 192.168.99.3 -t "uwb/anchor/+/vector"`

### **PGO Optimization Fails:**
```bash
❌ PGO optimization failed: No graph data generated
```
**Solution:**
- Need measurements from at least 1 anchor
- Wait for 1-second sliding window to fill
- Check that vector data contains valid x,y,z values

### **Position Outside Room:**
**Solution:**
- Check anchor orientations (45° facing into room)
- Verify room dimensions in `create_anchor_edges.py`
- Review coordinate system alignment

## 🎪 Testing Without Live Data

### **Use Past Data for Testing:**
```bash
python past_data.py
```
This runs PGO on CSV data to verify the system works.

### **Check Individual Components:**
```bash
python pgo_3d.py          # Test 3D PGO solver
python ingest_data.py     # Test data ingestion
python transform_to_global_vector.py  # Test coordinate transforms
```

## 🛑 Stopping the Demo

- **Graceful shutdown:** Close the matplotlib window
- **Force stop:** Press `Ctrl+C` in terminal
- **Cleanup:** All resources are automatically cleaned up

## 📈 Performance Expectations

- **Update Rate:** 1Hz position updates
- **Latency:** ~100ms from MQTT message to position update
- **Accuracy:** Depends on UWB measurement quality
- **Trajectory:** Smooth tracking with 100-point history

## 🎯 Success Indicators

✅ **System Working Correctly:**
- MQTT connection established
- UWB measurements streaming in
- Phone position updating every second
- Position stays within room bounds (0-440cm × 0-550cm)
- Trajectory shows smooth movement

🎉 **Ready for Live Testing!**

Move the phone around the room and watch the real-time 3D PGO tracking in action!
