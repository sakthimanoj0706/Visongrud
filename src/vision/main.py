import argparse
import cv2
import time
import mediapipe as mp
from ultralytics import YOLO
import collections

from src.vision.utils.api_client import send_frame, send_event
from src.vision.utils.io_utils import save_snapshot

# ---------------- CONFIG ----------------
MODE = "PEOPLE_MODE"

model = YOLO("models/yolov8n.pt")  
pose = mp.solutions.pose.Pose()

def run_people_camera(camera_id: int = 0) -> None:
    cam_str = f"CAM_PEOPLE_{camera_id:02d}"
    print(f"🔥 GuardianVision Engine started in {MODE} for {cam_str}")

    hip_y_history = collections.deque(maxlen=10)
    
    # ---------------- MAIN LOOP ----------------
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        results = model(frame, verbose=False)[0]
        
        alerts = []
        max_risk = 0
        snapshot_needed = False

        persons = []
        helmets = []
        items = []

        for box_data in results.boxes:
            cls = int(box_data.cls[0])
            label = model.names[cls]
            x1, y1, x2, y2 = map(int, box_data.xyxy[0])
            
            # Only draw relevant boxes to avoid noise
            if label in ["person", "cell phone", "car", "bus", "truck"] or "helmet" in label or "hat" in label:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if label == "person":
                persons.append((x1, y1, x2, y2))
            elif "helmet" in label or label == "hard-hat" or label == "hat":
                helmets.append((x1, y1, x2, y2))
            else:
                items.append({"label": label, "box": (x1, y1, x2, y2)})

        # ---------------- LAYER 2: PPE SAFETY ----------------
        for px1, py1, px2, py2 in persons:
            head_box_y2 = py1 + (py2 - py1) // 3
            wearing_helmet = False
            for hx1, hy1, hx2, hy2 in helmets:
                # Check overlap on upper 1/3 of body
                if hx1 > px1 - 20 and hx2 < px2 + 20 and hy2 < head_box_y2 + 20:
                    wearing_helmet = True
                    break
            
            if not wearing_helmet:
                alerts.append({"type": "PPE_NO_HELMET", "desc": "Worker missing helmet"})
                max_risk = max(max_risk, 30)

            # Assuming NO VEST for demonstration purposes, or could be mocked depending on user requirement
            alerts.append({"type": "PPE_NO_VEST", "desc": "Worker missing safety vest"})
            max_risk = max(max_risk, 30)

        # ---------------- LAYER 3: BEHAVIOUR-BASED SAFETY ----------------
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_res = pose.process(rgb)
        
        # 1. Fall detection using MediaPipe Pose
        if pose_res.pose_landmarks:
            hip = pose_res.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_HIP]
            hip_y = hip.y
            hip_y_history.append(hip_y)
            
            if len(hip_y_history) >= 5:
                # Sudden drop in Y
                diff = hip_y_history[-1] - hip_y_history[0]
                if diff > 0.15 and hip_y > 0.7:  
                    alerts.append({"type": "BBS_FALL_DETECTED", "desc": "Worker fall detected"})
                    max_risk = max(max_risk, 100)
                    snapshot_needed = True
                    
        # 2. Distraction (Phone near face)
        phone_present = any(i["label"] == "cell phone" for i in items)
        if phone_present and persons:
            alerts.append({"type": "BBS_DISTRACTION_PHONE", "desc": "Distraction via cell phone"})
            max_risk = max(max_risk, 50)
            snapshot_needed = True

        # 3. Unsafe proximity (near large vehicles/cranes)
        for p in persons:
            px1, py1, px2, py2 = p
            for item in items:
                if item["label"] in ["truck", "bus", "car"]:
                    ix1, iy1, ix2, iy2 = item["box"]
                    # 1D boundary overlap
                    if not (px2 < ix1 or px1 > ix2):
                        alerts.append({"type": "BBS_UNSAFE_PROXIMITY", "desc": f"Worker danger proximity to {item['label']}"})
                        max_risk = max(max_risk, 80)
                        snapshot_needed = True

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
                photo_path = save_snapshot(frame, prefix="people")

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
    parser = argparse.ArgumentParser(description="Run GuardianVision PEOPLE camera")
    parser.add_argument("--camera-id", type=int, default=1, help="Camera ID index (e.g. 1)")
    args = parser.parse_args()
    
    run_people_camera(camera_id=args.camera_id)
