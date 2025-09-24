# uwb_pgo.py
# Pose Graph Optimisation for 4-anchor UWB vectors -> iPhone trajectory (cm)
# Saves: CSV of optimised (x,y,z), a static PNG, and an animation (MP4 if ffmpeg exists, else GIF).

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import json, time, signal, sys
from datetime import datetime
import paho.mqtt.client as mqtt

# ====== mqtt configs ======
MQTT_BROKER = "192.168.99.3"
MQTT_PORT = 1883
MQTT_USERNAME = "laptop"
MQTT_PASSWORD = "laptop"
MQTT_CLIENT_ID = "uwb_pgo_subscriber"

MQTT_BASE_TOPIC = "uwb"
MQTT_QOS = 0
MQTT_KEEPALIVE = 60
WINDOW_MS = 120       # align anchors within 120 ms
MIN_ANCHORS = 1       # accept frame even if only 1 anchor arrived
MAX_FRAMES = 2000     # stop collecting after N frames (or Ctrl+C)

class LiveVectorCollector:
    def __init__(self):
        self.records = []
    def add(self, rec: dict):
        self.records.append(rec)


# =======mqtt connections=======
def _on_connect(client, userdata, flags, rc):
    if rc == 0:
        topic = f"{MQTT_BASE_TOPIC}/anchor/+/vector"
        client.subscribe(topic, qos=MQTT_QOS)
        print(f"✓ MQTT connected, subscribed: {topic}")
    else:
        print(f"✗ MQTT connect failed rc={rc}")


def on_disconnect(client, userdata, rc):
    print("✗ Disconnected from MQTT broker")

# def on_message(client, userdata, msg):
#     try:
#         topic_parts = msg.topic.split('/')
#         if len(topic_parts) >= 4 and topic_parts[-1] == "vector":
#             anchor_id = int(topic_parts[-2])
#             data = json.loads(msg.payload.decode('utf-8'))
#             data['_anchor_id'] = anchor_id
#             data['_received_time'] = time.time()
#             # -> hand off to your frame builder (added below)
#             userdata['ingest'](data)
#     except Exception as e:
#         print(f"✗ Error parsing message: {e}")
def _on_message(client, userdata, msg):
    try:
        parts = msg.topic.split('/')
        if len(parts) >= 4 and parts[-1] == 'vector' and parts[-3] == 'anchor':
            anchor_id = int(parts[-2])
        else:
            return
        recv_time_s = time.time()
        payload = json.loads(msg.payload.decode('utf-8'))
        rec = _record_from_msg(anchor_id, payload, recv_time_s)
        if rec is None:
            return  # skip if vector missing
        userdata['collector'].add(rec)
        print(rec, flush=True) #print each record as it arrives
        # If you prefer single-line JSON:
        # print(json.dumps(rec, ensure_ascii=False), flush=True)
    except Exception as e:
        print(f"✗ on_message error: {e}", flush=True)


# =========mqtt connection helpers========
def connect_mqtt(collector: LiveVectorCollector):
    # Pin to MQTT v3.1.1 so v1-style callbacks are OK (silences the deprecation warning)
    client = mqtt.Client(client_id=MQTT_CLIENT_ID, userdata={'collector': collector},
                         protocol=mqtt.MQTTv311)
    if MQTT_USERNAME and MQTT_PASSWORD:
        client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
    client.on_connect = _on_connect
    client.on_message = _on_message
    client.connect(MQTT_BROKER, MQTT_PORT, MQTT_KEEPALIVE)
    client.loop_start()
    return client


def disconnect_mqtt(client):
    client.loop_stop()
    client.disconnect()
# ========end of mqtt stuff========

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

# 45° “facing into the room” yaw for each board; local x points along heading into the room,
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

def load_vectors(csv_path: str):
    """Parse the CSV and return:
       kept_ts:   list of timestamp strings used,
       pred_pos:  list of arrays (k,3) giving per-anchor predicted phone pos for each timestep,
       """
    df = pd.read_csv(csv_path)

    # Only care about: timestamp_readable and anchor_n_local_{x,y,z}
    needed = ["timestamp_readable"]
    vec_cols = {}
    for n in range(4):
        cols = [f"anchor_{n}_local_x", f"anchor_{n}_local_y", f"anchor_{n}_local_z"]
        vec_cols[n] = cols
        for c in cols:
            if c not in df.columns:
                raise ValueError(f"Missing column: {c}")
        needed += cols

    # Maintain original order but keep only the needed columns
    df = df[[c for c in df.columns if c in needed]]

    kept_ts, pred_pos = [], []
    for i in range(len(df)):
        preds = []
        for n in range(4):
            x, y, z = df.loc[i, vec_cols[n][0]], df.loc[i, vec_cols[n][1]], df.loc[i, vec_cols[n][2]]
            if pd.notna(x) and pd.notna(y) and pd.notna(z):
                v_local = np.array([float(x), float(y), float(z)], dtype=float)     # cm
                v_global = ANCHOR_R[n] @ v_local                                   # rotate into global XY
                preds.append(ANCHORS[n] + v_global)                                # anchor + vector
        if len(preds) >= 1:  # tolerate missing anchors; at least one is enough
            kept_ts.append(str(df.loc[i, "timestamp_readable"]))
            pred_pos.append(np.vstack(preds))

    if not pred_pos:
        raise RuntimeError("No usable rows (need at least one complete anchor vector per row).")
    return kept_ts, pred_pos

def optimise(pred_pos):
    """Nonlinear least-squares over all timestamps; each row has 1..4 anchor->phone edges."""
    K = len(pred_pos)
    init_positions = np.vstack([pp.mean(axis=0) for pp in pred_pos])  # (K,3)
    x0 = init_positions.reshape(-1)

    def residuals(x):
        X = x.reshape(K, 3)
        errs = []
        for j in range(K):
            for p in pred_pos[j]:
                errs.append(X[j] - p)  # cm residuals in x,y,z
        return np.concatenate(errs)

    res = least_squares(residuals, x0)      # unconstrained; Levenberg–Marquardt-like (trf default)
    return res.x.reshape(K, 3), res.cost

def plot_static(X_opt: np.ndarray, out_png: str):
    fig, ax = plt.subplots(figsize=(7, 6))

    # grid
    for x in np.arange(0, ROOM_W + 1e-6, SQUARE_CM):
        ax.plot([x, x], [0, ROOM_H], linewidth=0.5)
    for y in np.arange(0, ROOM_H + 1e-6, SQUARE_CM):
        ax.plot([0, ROOM_W], [y, y], linewidth=0.5)

    # anchors
    ax.scatter([ANCHORS[i][0] for i in range(4)],
               [ANCHORS[i][1] for i in range(4)],
               s=40, label="Anchors")
    for idx in [0, 1, 2, 3]:
        ax.text(ANCHORS[idx][0], ANCHORS[idx][1], f"A{idx}", ha="left", va="bottom")

    # trajectory (top-down XY) - show as points for clustering visualization
    ax.scatter(X_opt[:, 0], X_opt[:, 1], s=20, alpha=0.7, label="Optimised positions")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-20, ROOM_W + 20); ax.set_ylim(-20, ROOM_H + 20)
    ax.set_xlabel("X (cm)"); ax.set_ylabel("Y (cm)")
    ax.set_title("iPhone trajectory (top-down)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)

def save_animation(X_opt: np.ndarray, out_mp4: str, fps: int):
    """Try MP4 with ffmpeg; fallback to GIF. Works even if K == 1."""
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    K = len(X_opt)
    # If we only have one frame, duplicate it so writers are happy
    if K == 1:
        X_anim = np.vstack([X_opt, X_opt])  # 2 frames
    else:
        X_anim = X_opt

    fig, ax = plt.subplots(figsize=(7, 6))
    # draw grid
    for x in np.arange(0, ROOM_W + 1e-6, SQUARE_CM):
        ax.plot([x, x], [0, ROOM_H], linewidth=0.5)
    for y in np.arange(0, ROOM_H + 1e-6, SQUARE_CM):
        ax.plot([0, ROOM_W], [y, y], linewidth=0.5)
    # anchors
    ax.scatter([ANCHORS[i][0] for i in range(4)],
               [ANCHORS[i][1] for i in range(4)], s=40)
    for idx in [0, 1, 2, 3]:
        ax.text(ANCHORS[idx][0], ANCHORS[idx][1], f"A{idx}", ha="left", va="bottom")

    path_points = ax.scatter([], [], s=20, alpha=0.7, label="Optimised positions")
    current_point, = ax.plot([], [], marker="o", markersize=8, color='red')  # Current position
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-20, ROOM_W + 20); ax.set_ylim(-20, ROOM_H + 20)
    ax.set_xlabel("X (cm)"); ax.set_ylabel("Y (cm)")
    ax.set_title("iPhone trajectory over time"); ax.legend()

    def init():
        path_points.set_offsets(np.empty((0, 2)))  # Empty array for scatter
        current_point.set_data([], [])
        return path_points, current_point

    def update(f):
        # Update scatter points for all positions up to current frame
        path_points.set_offsets(X_anim[:f+1])
        # Update current position marker
        current_point.set_data([X_anim[f, 0]], [X_anim[f, 1]])
        return path_points, current_point

    anim = animation.FuncAnimation(
        fig, update, init_func=init,
        frames=len(X_anim), interval=1000 / max(1, fps), blit=True
    )

    try:
        anim.save(out_mp4, writer="ffmpeg", fps=fps)
        saved_path = out_mp4
    except Exception:
        # fallback to GIF if ffmpeg is missing
        from matplotlib.animation import PillowWriter
        gif_path = os.path.splitext(out_mp4)[0] + ".gif"
        anim.save(gif_path, writer=PillowWriter(fps=fps))
        saved_path = gif_path

    plt.close(fig)
    return saved_path

def _record_from_msg(anchor_id: int, data: dict, recv_time_s: float):
    # timestamp: prefer device t_unix_ns, else received time
    if 't_unix_ns' in data and isinstance(data['t_unix_ns'], (int, float)):
        ts_ns = int(data['t_unix_ns'])
        ts_readable = datetime.fromtimestamp(ts_ns / 1e9).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    else:
        ts_readable = datetime.fromtimestamp(recv_time_s).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    v = data.get('vector_local') or {}
    return {
        "timestamp": ts_readable,
        "anchor_id": int(anchor_id),
        "local_vector_x": float(v.get('x', float('nan'))),
        "local_vector_y": float(v.get('y', float('nan'))),
        "local_vector_z": float(v.get('z', float('nan'))),
    }


def main():
    ap = argparse.ArgumentParser()
    # Either CSV or MQTT, not both
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", help="Path to uwb_data_*.csv")
    src.add_argument("--mqtt", action="store_true",
                    help="Read vectors live from MQTT instead of CSV")
    args = ap.parse_args()

    if args.mqtt:
        collector = LiveVectorCollector()
        client = connect_mqtt(collector)
        print("▶ Reading live MQTT vectors… Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        disconnect_mqtt(client)


    # csv_path = args.csv
    # if not os.path.isfile(csv_path):
    #     raise FileNotFoundError(f"CSV not found: {csv_path}")

    # os.makedirs(args.outdir, exist_ok=True)
    # kept_ts, pred_pos = load_vectors(csv_path)
    # X_opt, cost = optimise(pred_pos)

    # # Save CSV
    # out_csv = os.path.join(args.outdir, "iphone_trajectory_optimised.csv")
    # pd.DataFrame({
    #     "timestamp_readable": kept_ts,
    #     "x_cm": X_opt[:, 0],
    #     "y_cm": X_opt[:, 1],
    #     "z_cm": X_opt[:, 2],
    # }).to_csv(out_csv, index=False)

    # # Save static and animation
    # out_png = os.path.join(args.outdir, "iphone_trajectory_static.png")
    # plot_static(X_opt, out_png)
    # out_mp4 = os.path.join(args.outdir, "iphone_trajectory_animation.mp4")
    # out_anim = save_animation(X_opt, out_mp4, fps=args.fps)

    # print(f"\nOptimisation frames: {len(X_opt)}")
    # print(f"Residual cost (sum of squared cm errors): {cost:.3f}")
    # print(f"Saved:\n  {out_csv}\n  {out_png}\n  {out_anim}")

if __name__ == "__main__":
    main()
