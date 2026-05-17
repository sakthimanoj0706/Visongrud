import os
from pathlib import Path
from typing import Optional
from datetime import datetime, timedelta
import sqlite3
import base64
import cv2
import numpy as np

from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from src.notifications.notifier import (
    is_critical_event,
    send_realtime_alert,
    send_email_report
)

app = FastAPI()

Path("data").mkdir(exist_ok=True)
Path("data/snapshots").mkdir(exist_ok=True)

DB_PATH = "data/events.db"
LATEST_FRAME = None


def get_db_connection():
    try:
        conn = sqlite3.connect(DB_PATH)
        return conn
    except Exception as e:
        print(f"[DB-ERROR] Failed to connect to database at {DB_PATH}: {e}")
        return None

class Event(BaseModel):
    camera_id: str
    type: str
    risk: int
    description: str
    photo_path: Optional[str] = ""


@app.post("/event")
def log_event(event: Event, background_tasks: BackgroundTasks):
    conn = get_db_connection()
    if conn is not None:
        try:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS safety_events(
                    time TEXT,
                    camera_id TEXT,
                    type TEXT,
                    risk INT,
                    description TEXT,
                    photo_path TEXT
                )
            """)
            event_time = datetime.now().isoformat()
            cur.execute("INSERT INTO safety_events VALUES (?,?,?,?,?,?)",
                        (event_time, event.camera_id, event.type, event.risk, event.description, event.photo_path))
            conn.commit()
        except Exception as e:
            print(f"[DB-ERROR] Failed to insert event: {e}")
        finally:
            conn.close()
    
    event_dict = event.dict()
    # Provide time key properly
    event_dict["time"] = datetime.now().isoformat()
    if is_critical_event(event_dict):
        background_tasks.add_task(send_realtime_alert, event_dict)

    return {"status": "logged"}


@app.post("/frame")
def upload_frame(payload: dict):
    global LATEST_FRAME
    img_b64 = payload.get("image_base64") or payload.get("image")
    if not img_b64:
        return {"error": "Missing image_base64 or image field"}
    try:
        img = base64.b64decode(img_b64)
        npimg = np.frombuffer(img, np.uint8)
        LATEST_FRAME = cv2.imdecode(npimg, 1)
        return {"ok": True}
    except Exception as e:
        return {"error": f"Failed to decode image: {e}"}


@app.get("/frame")
def get_frame():
    if LATEST_FRAME is None:
        return {"image_base64": None}
    _, buf = cv2.imencode(".jpg", LATEST_FRAME)
    return {"image_base64": base64.b64encode(buf).decode()}


@app.get("/events")
def get_events(limit: int = 50):
    conn = get_db_connection()
    if conn is None:
        return []
    
    events = []
    try:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS safety_events(
                time TEXT,
                camera_id TEXT,
                type TEXT,
                risk INT,
                description TEXT,
                photo_path TEXT
            )
        """)
        cur.execute("SELECT * FROM safety_events ORDER BY time DESC LIMIT ?", (limit,))
        rows = cur.fetchall()
        for r in rows:
            events.append({
                "time": r[0],
                "camera_id": r[1],
                "type": r[2],
                "risk": r[3],
                "description": r[4],
                "photo_path": r[5]
            })
    except Exception as e:
        print(f"[DB-ERROR] Failed to fetch events: {e}")
    finally:
        conn.close()
        
    return {"events": events}


@app.get("/email-report")
def get_email_report():
    conn = get_db_connection()
    if conn is None:
        return {"status": "error", "message": "DB connection failed"}
        
    try:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS safety_events(
                time TEXT,
                camera_id TEXT,
                type TEXT,
                risk INT,
                description TEXT,
                photo_path TEXT
            )
        """)
        now = datetime.now()
        yesterday = (now - timedelta(hours=24)).isoformat()
        
        cur.execute("""
            SELECT camera_id, type, COUNT(*), MAX(risk)
            FROM safety_events
            WHERE time >= ?
            GROUP BY camera_id, type
        """, (yesterday,))
        rows = cur.fetchall()
    except Exception as e:
        print(f"[DB-ERROR] Failed to generate email report: {e}")
        rows = []
    finally:
        conn.close()
        
    report_lines = ["GuardianVision Daily Safety Report\n"]
    report_lines.append(f"Time window: {yesterday} to {datetime.now().isoformat()}\n")
    
    cam_group = {}
    for r in rows:
        cam_id, ev_type, count, max_risk = r[0], r[1], r[2], r[3]
        if cam_id not in cam_group:
            cam_group[cam_id] = []
        cam_group[cam_id].append((ev_type, count, max_risk))
        
    for cam_id, evs in cam_group.items():
        report_lines.append(f"{cam_id}:")
        for ev in evs:
            report_lines.append(f"  - {ev[0]}: {ev[1]} events (max risk {ev[2]})")
        report_lines.append("")
        
    body = "\n".join(report_lines)
    send_email_report("GuardianVision Daily Safety Report", body)
    
    return {"status": "sent", "body": body}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.server:app", reload=True)
