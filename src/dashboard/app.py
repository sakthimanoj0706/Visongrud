import streamlit as st
import requests
import base64
import cv2
import numpy as np
import time

API = "http://127.0.0.1:8000"

st.set_page_config(layout="wide")
st.title("🛡 GuardianVision Live Dashboard")

col1, col2 = st.columns([3, 2])

frame_box = col1.empty()
risk_bar_box = col2.empty()

with col2:
    tab1, tab2 = st.tabs(["Layer 1 – Zone Security (ENTRY)", "Layers 2+3 – PPE + BBS (PEOPLE)"])
    with tab1:
        entry_box = st.empty()
    with tab2:
        people_box = st.empty()

while True:
    # 1) Live camera feed
    try:
        resp = requests.get(f"{API}/frame", timeout=1)
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, dict) and "error" in data:
                frame_box.info(data["error"])
            else:
                # data is expected to be a JSON with image_base64
                img64 = data.get("image_base64")
                if img64:
                    img = base64.b64decode(img64)
                    arr = np.frombuffer(img, np.uint8)
                    frame = cv2.imdecode(arr, 1)
                    if frame is not None:
                        # Streamlit expects RGB but cv2 uses BGR
                        frame_box.image(frame, channels="BGR")
                else:
                    frame_box.info("No frame available")
    except Exception as e:
        frame_box.error(f"Error fetching frame or camera offline")

    # 2) Events & Risk
    try:
        events_resp = requests.get(f"{API}/events?limit=50", timeout=1)
        if events_resp.status_code == 200:
            events_data = events_resp.json()
            events = events_data.get("events", [])
            
            if events:
                max_risk = max([e["risk"] for e in events])
                risk_bar_box.progress(min(max_risk, 100), text=f"Max Recent Risk: {max_risk}")
            else:
                risk_bar_box.progress(0, text="Max Recent Risk: 0")

            entry_events = []
            people_events = []
            for e in events:
                cam_id = e["camera_id"]
                e_type = e["type"]
                if cam_id.startswith("CAM_ENTRY") or e_type.startswith("ENTRY_"):
                    entry_events.append(e)
                elif cam_id.startswith("CAM_PEOPLE") or e_type.startswith("PPE_") or e_type.startswith("BBS_"):
                    people_events.append(e)

            def render_events(box, ev_list):
                with box.container():
                    if not ev_list:
                        st.info("No events in this category yet.")
                    for ev in ev_list:
                        st.markdown(f"**{ev['time']}** | **{ev['type']}** | Risk: **{ev['risk']}**")
                        st.write(ev["description"])
                        if ev["photo_path"]:
                            try:
                                img_cv = cv2.imread(ev["photo_path"])
                                if img_cv is not None:
                                    st.image(img_cv, channels="BGR", use_container_width=True)
                                else:
                                    st.write(f"(Image not found: {ev['photo_path']})")
                            except Exception:
                                st.write(f"(Image error: {ev['photo_path']})")
                        else:
                            st.write("(No image)")
                        st.divider()

            render_events(entry_box, entry_events)
            render_events(people_box, people_events)
            
    except Exception as e:
        risk_bar_box.error("Backend offline or error fetching events.")

    time.sleep(0.5)
