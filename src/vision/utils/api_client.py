import requests
import numpy as np
from src.vision.utils.io_utils import frame_to_base64

API_URL = "http://127.0.0.1:8000"

def send_frame(frame: np.ndarray) -> None:
    try:
        b64 = frame_to_base64(frame)
        payload = {"image_base64": b64}
        requests.post(f"{API_URL}/frame", json=payload, timeout=1)
    except Exception as e:
        print(f"[API-ERROR] Failed to send frame: {e}")

def send_event(camera_id: str, event_type: str, risk: int, desc: str, photo_path: str | None = None) -> None:
    event_payload = {
        "camera_id": camera_id,
        "type": event_type,
        "risk": risk,
        "description": desc,
        "photo_path": photo_path or ""
    }
    
    try:
        requests.post(f"{API_URL}/event", json=event_payload, timeout=1)
    except Exception as e:
        print(f"[API-ERROR] Failed to send event: {e}")
