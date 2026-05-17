import cv2
import base64
import time
import os
import numpy as np
from pathlib import Path

# Ensure snapshots folder exists (also done by backend, but safe here)
Path("data/snapshots").mkdir(parents=True, exist_ok=True)

def frame_to_base64(frame: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", frame)
    return base64.b64encode(buf).decode()

def save_snapshot(frame: np.ndarray, prefix: str) -> str:
    timestamp = int(time.time())
    fname = f"data/snapshots/{prefix}_{timestamp}.jpg"
    try:
        cv2.imwrite(fname, frame)
    except Exception as e:
        print(f"[IO-ERROR] Failed to write snapshot: {e}")
    return fname
