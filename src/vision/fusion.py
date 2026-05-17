import argparse
import cv2
import time
from ultralytics import YOLO
from shapely.geometry import box
from datetime import datetime
import numpy as np

from src.vision.utils.api_client import send_frame, send_event
from src.vision.utils.io_utils import save_snapshot
from src.vision.utils.geometry_utils import load_zone_polygon

# ---------------- CONFIG ----------------
MODE = "ENTRY_MODE"
AFTER_HOURS_START = 18 # 6 PM
AFTER_HOURS_END = 6    # 6 AM

model = YOLO("models/yolov8n.pt")  

def run_entry_camera(camera_id: int = 0) -> None:
    cam_str = f"CAM_ENTRY_{camera_id:02d}"
    print(f"🔥 GuardianVision Engine started in {MODE} for {cam_str}")
    
    restricted_zone = load_zone_polygon(camera_id, MODE)

    # ---------------- MAIN LOOP ----------------
    cap = cv2.VideoCapture(0)
    last_seen = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        results = model(frame, verbose=False)[0]
        
        alerts = []
        max_risk = 0
        snapshot_needed = False
        persons = []
        
        # Layer 1 detection
        for box_data in results.boxes:
            cls = int(box_data.cls[0])
            label = model.names[cls]
            
            if label == "person":
                x1, y1, x2, y2 = map(int, box_data.xyxy[0])
                persons.append((x1, y1, x2, y2))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
        # Zone & time logic
        hour = datetime.now().hour
        is_after_hours = (hour >= AFTER_HOURS_START or hour < AFTER_HOURS_END)

        if persons:
            last_seen = time.time()
            for p in persons:
                x1, y1, x2, y2 = p
                bb = box(x1, y1, x2, y2)
                if restricted_zone.intersects(bb):
                    alerts.append({"type": "ENTRY_RESTRICTED_ZONE", "desc": "Person in restricted zone"})
                    max_risk = max(max_risk, 90)
                    snapshot_needed = True

            if is_after_hours:
                alerts.append({"type": "ENTRY_AFTER_HOURS", "desc": "Person detected after hours"})
                max_risk = max(max_risk, 95)
                snapshot_needed = True
        else:
            if time.time() - last_seen > 5.0:
                alerts.append({"type": "ENTRY_NO_PERSON", "desc": "No person visible for >5s"})
                max_risk = max(max_risk, 20)

        # Draw Zone
        pts = np.array(restricted_zone.exterior.coords, np.int32)
        cv2.polylines(frame, [pts], True, (0, 0, 255), 2)
        cv2.putText(frame, "RESTRICTED", pts[0], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        send_frame(frame)
        
        if alerts:
            # Dedup
            unique_types = []
            unique_desc = []
            seen = set()
            for a in alerts:
                if a["type"] not in seen:
                    unique_types.append(a["type"])
                    unique_desc.append(a["desc"])
                    seen.add(a["type"])
                    
            combined_type = ";".join(unique_types)
            combined_desc = ", ".join(unique_desc)

            photo_path = ""
            if snapshot_needed:
                photo_path = save_snapshot(frame, prefix="entry")

            send_event(
                camera_id=cam_str,
                event_type=combined_type,
                risk=max_risk,
                desc=combined_desc,
                photo_path=photo_path
            )
            print(f"\nEVENTS: {combined_type} (Risk: {max_risk})")

        cv2.imshow(f"{MODE} Dashboard", frame)
        if cv2.waitKey(1) == 27: break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GuardianVision ENTRY camera")
    parser.add_argument("--camera-id", type=int, default=1, help="Camera ID index (e.g. 1)")
    args = parser.parse_args()
    
    run_entry_camera(camera_id=args.camera_id)
