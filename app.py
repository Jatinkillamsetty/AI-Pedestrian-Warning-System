"""
AI-Based Pedestrian Detection and Heavy-Vehicle Blind-Zone Warning System
Main Streamlit Application Dashboard with ESP32 Hardware Integration.
"""

import streamlit as st
import cv2
import numpy as np
import time

from config import (
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_ALERT_COOLDOWN,
    DEFAULT_AUDIO_ENABLED,
    DEFAULT_VOICE_TYPE,
    ZONE_PRESETS,
    ESP32_PIN_MAP,
    ULTRASONIC_DANGER_THRESHOLD_CM
)
from camera import CameraManager
from detector import PedestrianDetector
from danger_zone import DangerZone
from alert_manager import AlertManager
from esp32_interface import ESP32Interface
from utils.fps import FPSCounter
from utils.drawing import draw_detections, draw_hud_banner

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Heavy-Vehicle Blind-Zone Safety AI",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern Industrial Safety Dark Theme
st.markdown("""
<style>
    /* Dark Theme Custom Palette */
    .stApp {
        background-color: #0F172A;
        color: #F8FAFC;
    }
    
    /* Top Header Styling */
    .main-header {
        background: linear-gradient(90deg, #1E293B 0%, #0F172A 100%);
        padding: 18px 24px;
        border-radius: 12px;
        border-left: 6px solid #3B82F6;
        margin-bottom: 20px;
    }
    
    .main-title {
        font-size: 26px;
        font-weight: 800;
        color: #F8FAFC;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .sub-title {
        font-size: 14px;
        color: #94A3B8;
        margin-top: 4px;
        margin-bottom: 0;
    }
    
    /* Metric Cards */
    .metric-box {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 14px;
        text-align: center;
    }
    
    .metric-label {
        font-size: 11px;
        font-weight: 600;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-value-safe {
        font-size: 18px;
        font-weight: 700;
        color: #10B981;
    }
    
    .metric-value-danger {
        font-size: 18px;
        font-weight: 700;
        color: #EF4444;
    }

    /* Live Video Container */
    .video-container {
        border: 2px solid #334155;
        border-radius: 12px;
        overflow: hidden;
        background-color: #000000;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if "camera_manager" not in st.session_state:
    st.session_state.camera_manager = CameraManager()

if "detector" not in st.session_state:
    st.session_state.detector = PedestrianDetector()

if "danger_zone" not in st.session_state:
    st.session_state.danger_zone = DangerZone(preset_name="Front Blind Spot")

if "alert_manager" not in st.session_state:
    st.session_state.alert_manager = AlertManager()

if "esp32_interface" not in st.session_state:
    st.session_state.esp32_interface = ESP32Interface()
    st.session_state.esp32_interface.connect("SIMULATOR")

if "fps_counter" not in st.session_state:
    st.session_state.fps_counter = FPSCounter()

if "is_running" not in st.session_state:
    st.session_state.is_running = False

if "detection_enabled" not in st.session_state:
    st.session_state.detection_enabled = True

if "ultrasonic_fallback_enabled" not in st.session_state:
    st.session_state.ultrasonic_fallback_enabled = False


# Shortcuts
cam_mgr = st.session_state.camera_manager
detector = st.session_state.detector
zone = st.session_state.danger_zone
alert_mgr = st.session_state.alert_manager
esp32 = st.session_state.esp32_interface
fps_counter = st.session_state.fps_counter

# ---------------------------------------------------------
# Header & Disclaimer
# ---------------------------------------------------------
st.markdown("""
<div class="main-header">
    <div class="main-title">🚛 HEAVY-VEHICLE BLIND-ZONE AI SYSTEM + ESP32 HARDWARE</div>
    <div class="sub-title">Computer Vision AI + ESP32 Hardware Integration (Red/Green LEDs, HC-SR04 Sensor)</div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Sidebar Controls & Configuration
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙ System Controls")
    
    # --- Tab 1: ESP32 Hardware Controls ---
    st.subheader("🔌 ESP32 Hardware Interface")
    available_ports = esp32.list_available_ports()
    selected_port = st.selectbox("Serial Port", options=available_ports, index=0)
    selected_baud = st.selectbox("Baud Rate", options=[115200, 9600, 57600], index=0)

    col_hw1, col_hw2 = st.columns(2)
    with col_hw1:
        if st.button("🔌 Connect ESP32", use_container_width=True, type="primary"):
            res = esp32.connect(selected_port, selected_baud)
            if res:
                st.toast(f"Connected to ESP32: {selected_port}", icon="✅")
            else:
                st.toast("Opened Hardware Simulator Mode", icon="ℹ")

    with col_hw2:
        if st.button("🔌 Disconnect", use_container_width=True):
            esp32.disconnect()
            st.toast("Hardware disconnected.", icon="🛑")

    mode_str = "SIMULATED HARDWARE" if esp32.use_simulation else ("PHYSICAL ESP32" if esp32.is_connected else "DISCONNECTED")
    mode_color = "#3B82F6" if esp32.use_simulation else ("#10B981" if esp32.is_connected else "#EF4444")
    st.caption(f"Hardware Status: <span style='color:{mode_color}; font-weight:bold;'>{mode_str}</span>", unsafe_allow_html=True)

    st.session_state.ultrasonic_fallback_enabled = st.checkbox(
        "Enable Ultrasonic Sensor Alert Override",
        value=False,
        help="If checked, ultrasonic readings < 30cm will also trigger danger even without AI person detection."
    )

    with st.expander("📌 ESP32 Circuit Pinout Reference"):
        st.markdown(f"""
        - **Green LED**: `GPIO {ESP32_PIN_MAP['GREEN_LED']}` (SAFE state)
        - **Red LED**: `GPIO {ESP32_PIN_MAP['RED_LED']}` (DANGER state)
        - **HC-SR04 Trig**: `GPIO {ESP32_PIN_MAP['TRIG']}`
        - **HC-SR04 Echo**: `GPIO {ESP32_PIN_MAP['ECHO']}`
        """)

    st.markdown("---")

    # --- Tab 2: Camera Controls ---
    st.subheader("📷 Camera Settings")
    cam_index = st.number_input("Webcam Index", min_value=0, max_value=5, value=0, step=1)
    
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("▶ Start Camera", use_container_width=True, type="primary"):
            success = cam_mgr.start(camera_index=cam_index)
            if success:
                st.session_state.is_running = True
                st.toast("Webcam connected!", icon="✅")
            else:
                st.error("Camera not detected! Click 'Enable Demo Mode' below.")
    
    with col_btn2:
        if st.button("■ Stop Camera", use_container_width=True):
            cam_mgr.stop()
            st.session_state.is_running = False
            st.toast("Camera stopped.", icon="🛑")
            
    if st.button("🎭 Enable Demo Video Stream", use_container_width=True):
        cam_mgr.enable_demo_mode()
        st.session_state.is_running = True
        st.toast("Demo Video Mode Active!", icon="🎭")

    st.markdown("---")
    
    # --- Tab 3: Blind Zone Configuration ---
    st.subheader("🚧 Blind Zone Configuration")
    preset_choice = st.selectbox(
        "Zone Preset",
        options=list(ZONE_PRESETS.keys()),
        index=0
    )
    
    if preset_choice != zone.preset_name:
        zone.set_preset(preset_choice)

    if preset_choice == "Custom Rectangular":
        st.caption("Adjust Custom Zone Percentage Bounds:")
        x_min = st.slider("Left Boundary (X Min)", 0.0, 0.5, 0.15, 0.05)
        x_max = st.slider("Right Boundary (X Max)", 0.5, 1.0, 0.85, 0.05)
        y_min = st.slider("Top Boundary (Y Min)", 0.0, 0.8, 0.30, 0.05)
        y_max = st.slider("Bottom Boundary (Y Max)", 0.2, 1.0, 1.00, 0.05)
        zone.set_custom_rectangle(x_min, x_max, y_min, y_max)

    st.markdown("---")
    
    # --- Tab 4: AI & Detection Settings ---
    st.subheader("🧠 AI Detection Settings")
    st.session_state.detection_enabled = st.checkbox("Enable AI Person Detection", value=True)
    
    conf_thresh = st.slider(
        "Confidence Threshold",
        min_value=0.30,
        max_value=0.95,
        value=DEFAULT_CONF_THRESHOLD,
        step=0.05
    )
    detector.set_confidence(conf_thresh)
    
    st.caption(f"YOLO Model: `yolov8n.pt` | Device: `{detector.device.upper()}`")

    st.markdown("---")
    
    # --- Tab 5: Audio Warning Settings ---
    st.subheader("🔊 Audio Warning Settings")
    alert_mgr.audio_enabled = st.checkbox("Enable Speaker Audio Warning", value=DEFAULT_AUDIO_ENABLED)
    
    alert_mgr.cooldown_seconds = st.slider(
        "Alert Cooldown (Seconds)",
        min_value=1.0,
        max_value=10.0,
        value=DEFAULT_ALERT_COOLDOWN,
        step=0.5
    )
    
    alert_mgr.voice_type = st.selectbox(
        "Alert Sound Type",
        options=["Voice Warning", "Siren Beep"],
        index=0
    )
    
    if st.button("🔊 Test Audio Alert", use_container_width=True):
        alert_mgr.play_sound()
        esp32.send_alert_state(True)
        st.toast("Testing Speaker Sound & ESP32 Red LED...", icon="🔊")


# ---------------------------------------------------------
# Top Telemetry & Status Cards
# ---------------------------------------------------------
col_stat1, col_stat2, col_stat3, col_stat4, col_stat5, col_stat6 = st.columns(6)

status_placeholder = col_stat1.empty()
people_placeholder = col_stat2.empty()
danger_placeholder = col_stat3.empty()
dist_placeholder = col_stat4.empty()
hw_led_placeholder = col_stat5.empty()
fps_placeholder = col_stat6.empty()

# Render initial placeholders
status_placeholder.markdown("""
<div class="metric-box">
    <div class="metric-label">System Status</div>
    <div class="metric-value-safe">✓ AREA SAFE</div>
</div>
""", unsafe_allow_html=True)

people_placeholder.markdown("""
<div class="metric-box">
    <div class="metric-label">People Detected</div>
    <div style="font-size: 18px; font-weight: 700;">0</div>
</div>
""", unsafe_allow_html=True)

danger_placeholder.markdown("""
<div class="metric-box">
    <div class="metric-label">In Danger Zone</div>
    <div style="font-size: 18px; font-weight: 700;">0</div>
</div>
""", unsafe_allow_html=True)

dist_placeholder.markdown("""
<div class="metric-box">
    <div class="metric-label">HC-SR04 Distance</div>
    <div style="font-size: 18px; font-weight: 700; color: #94A3B8;">-- cm</div>
</div>
""", unsafe_allow_html=True)

hw_led_placeholder.markdown("""
<div class="metric-box">
    <div class="metric-label">ESP32 Hardware</div>
    <div style="font-size: 14px; font-weight: 700; color: #10B981;">🟢 GREEN LED</div>
</div>
""", unsafe_allow_html=True)

fps_placeholder.markdown("""
<div class="metric-box">
    <div class="metric-label">System FPS</div>
    <div style="font-size: 18px; font-weight: 700;">0.0</div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Main Live Camera Section
# ---------------------------------------------------------
st.subheader("📹 LIVE CAMERA & AI BLIND-ZONE OVERLAY")
video_placeholder = st.empty()


# If not running initially, display camera start prompt frame
if not st.session_state.is_running:
    err_frame = cam_mgr._generate_error_frame()
    video_placeholder.image(cv2.cvtColor(err_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
    st.info("💡 Click **▶ Start Camera** or **🎭 Enable Demo Video Stream** in the sidebar to begin live monitoring.")


# ---------------------------------------------------------
# Real-Time Frame Processing Loop
# ---------------------------------------------------------
while st.session_state.is_running:
    success, frame = cam_mgr.read_frame()
    if not success or frame is None:
        video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
        time.sleep(0.5)
        continue

    # 1. AI Person Detection & Danger Zone Evaluation
    detections = []
    is_ai_danger = False

    if st.session_state.detection_enabled:
        detections, is_ai_danger = detector.detect(frame, danger_zone=zone)

    # 2. Read ESP32 Hardware Sensor Telemetry (HC-SR04 Ultrasonic Distance)
    telemetry = esp32.read_telemetry()
    hcsr04_dist = telemetry.get("distance_cm", -1.0)
    
    # Check if ultrasonic sensor fallback override is explicitly enabled by user
    is_hw_danger = False
    if st.session_state.ultrasonic_fallback_enabled:
        is_hw_danger = (hcsr04_dist > 0.0 and hcsr04_dist <= ULTRASONIC_DANGER_THRESHOLD_CM)

    # Master Danger State: Controlled primarily by AI Person Detection in the Blind Zone!
    is_master_danger = is_ai_danger or is_hw_danger

    # 3. Audio Alert & ESP32 Hardware Signal Update (Red/Green LED)
    # Explicitly sends DANGER\n or SAFE\n to ESP32 over serial
    alert_mgr.update(is_master_danger, esp32_interface=esp32)

    # 4. Counts
    total_people = len(detections)
    danger_people_count = sum(1 for d in detections if d["is_in_danger"])

    # 5. Render OpenCV Overlays
    annotated_frame = zone.draw(frame, is_danger=is_master_danger)

    if st.session_state.detection_enabled:
        annotated_frame = draw_detections(annotated_frame, detections)

    current_fps = fps_counter.update()
    annotated_frame = draw_hud_banner(
        annotated_frame,
        is_danger=is_master_danger,
        people_count=total_people,
        danger_count=danger_people_count,
        fps=current_fps
    )

    # 6. Convert frame to RGB for Streamlit display
    rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    video_placeholder.image(rgb_frame, use_container_width=True)

    # 7. Format HC-SR04 Distance Display (Real Hardware Telemetry Reading)
    if hcsr04_dist > 0.0:
        dist_str = f"{hcsr04_dist:.1f} cm"
        dist_color = "#F59E0B" if (hcsr04_dist <= ULTRASONIC_DANGER_THRESHOLD_CM) else "#10B981"
    else:
        dist_str = "INVALID (Sensor Error)"
        dist_color = "#94A3B8"  # Slate/Gray text for sensor error

    # 8. Update Telemetry Cards
    if is_master_danger:
        status_placeholder.markdown("""
        <div class="metric-box" style="border: 2px solid #EF4444; background: #450A0A;">
            <div class="metric-label">System Status</div>
            <div class="metric-value-danger">⚠ WARNING: DANGER ZONE</div>
        </div>
        """, unsafe_allow_html=True)
        hw_led_placeholder.markdown("""
        <div class="metric-box" style="border: 1px solid #EF4444;">
            <div class="metric-label">ESP32 Hardware</div>
            <div style="font-size: 13px; font-weight: 700; color: #EF4444;">🔴 RED LED ACTIVE</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        status_placeholder.markdown("""
        <div class="metric-box">
            <div class="metric-label">System Status</div>
            <div class="metric-value-safe">✓ AREA SAFE</div>
        </div>
        """, unsafe_allow_html=True)
        hw_led_placeholder.markdown("""
        <div class="metric-box">
            <div class="metric-label">ESP32 Hardware</div>
            <div style="font-size: 13px; font-weight: 700; color: #10B981;">🟢 GREEN LED ACTIVE</div>
        </div>
        """, unsafe_allow_html=True)

    people_placeholder.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">People Detected</div>
        <div style="font-size: 18px; font-weight: 700;">{total_people}</div>
    </div>
    """, unsafe_allow_html=True)

    danger_placeholder.markdown(f"""
    <div class="metric-box" style="border: 1px solid {'#EF4444' if danger_people_count > 0 else '#334155'};">
        <div class="metric-label">In Danger Zone</div>
        <div style="font-size: 18px; font-weight: 700; color: {'#EF4444' if danger_people_count > 0 else '#F8FAFC'};">{danger_people_count}</div>
    </div>
    """, unsafe_allow_html=True)

    dist_placeholder.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">HC-SR04 Distance</div>
        <div style="font-size: 16px; font-weight: 700; color: {dist_color};">{dist_str}</div>
    </div>
    """, unsafe_allow_html=True)

    fps_placeholder.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">System FPS</div>
        <div style="font-size: 18px; font-weight: 700;">{current_fps:.1f}</div>
    </div>
    """, unsafe_allow_html=True)

    time.sleep(0.01)
