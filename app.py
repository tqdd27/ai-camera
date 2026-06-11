import os
# 【核心修復】強制讓 MediaPipe 在沒有螢幕的雲端 Linux 伺服器中順利執行
os.environ["XDG_RUNTIME_DIR"] = "/tmp/runtime-root"

import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
import numpy as np
import mediapipe as mp

# --- 官方標準多媒體處理解決方案導入 ---
mp_solutions = mp.solutions
FaceMesh = mp_solutions.face_mesh.FaceMesh
Hands = mp_solutions.hands.Hands

# --- 設定 ---
st.title("像個偉(偽)人一樣 📸")
st.write("👉 Thumbs up for [Qin Shihuang], open mouth for [Einstein]. Press Capture to take a photo!")

if "history" not in st.session_state:
    st.session_state.history = []

@st.cache_data
def load_resources():
    hair = cv2.imread("hair.png", cv2.IMREAD_UNCHANGED)
    hat = cv2.imread("hat.png", cv2.IMREAD_UNCHANGED)
    bear = cv2.imread("bear.png", cv2.IMREAD_UNCHANGED)
    return hair, hat, bear

hair_img, hat_img, bear_img = load_resources()

def overlay_image(background, overlay, x, y, size=None):
    if overlay is None: return background
    bg_h, bg_w = background.shape[:2]
    if size is not None:
        overlay = cv2.resize(overlay, size, interpolation=cv2.INTER_AREA)
    h, w = overlay.shape[:2]
    if x >= bg_w or y >= bg_h or x + w <= 0 or y + h <= 0: return background
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(bg_w, x + w), min(bg_h, y + h)
    overlay_x1, overlay_y1 = x1 - x, y1 - y
    overlay_x2, overlay_y2 = overlay_x1 + (x2 - x1), overlay_y1 + (y2 - y1)
    crop_overlay = overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2]
    crop_bg = background[y1:y2, x1:x2]
    if crop_overlay.shape[2] == 4:
        alpha = crop_overlay[:, :, 3] / 255.0
        alpha = np.expand_dims(alpha, axis=2)
        composite = crop_overlay[:, :, :3] * alpha + crop_bg * (1 - alpha)
        background[y1:y2, x1:x2] = composite
    else:
        background[y1:y2, x1:x2] = crop_overlay[:, :, :3]
    return background

class VideoProcessor:
    def __init__(self):
        # 使用官方路徑初始化
        self.face_mesh = FaceMesh(max_num_faces=1, refine_landmarks=False, min_detection_confidence=0.5)
        self.hands = Hands(max_num_hands=1, min_detection_confidence=0.5)
        self.latest_orig = None
        self.latest_filter = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.latest_orig = img.copy()
        
        h, w, _ = img.shape
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        face_results = self.face_mesh.process(rgb_img)
        hand_results = self.hands.process(rgb_img)
        
        status_text = "Scanning... Make a gesture!"
        
        if face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0]
            upper_lip = face_landmarks.landmark[13]
            lower_lip = face_landmarks.landmark[14]
            forehead = face_landmarks.landmark[10]
            chin = face_landmarks.landmark[152]
            
            lip_dist = abs(upper_lip.y - lower_lip.y) * h
            face_height = abs(forehead.y - chin.y) * h
            
            if lip_dist > (face_height * 0.15):
                status_text = "ACTIVE: Einstein Mode"
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                if hair_img is not None:
                    hair_w = int(face_height * 1.6)
                    hair_h = int(hair_w * (hair_img.shape[0] / hair_img.shape[1]))
                    img = overlay_image(img, hair_img, int(forehead.x * w - hair_w / 2), int(forehead.y * h - hair_h * 0.8), size=(hair_w, hair_h))

        if hand_results and hand_results.multi_hand_landmarks:
            hand_landmarks = hand_results.multi_hand_landmarks[0]
            thumb_tip = hand_landmarks.landmark[4]
            thumb_ip = hand_landmarks.landmark[3]
            index_mcp = hand_landmarks.landmark[5]
            
            if thumb_tip.y < thumb_ip.y and thumb_tip.y < index_mcp.y:
                status_text = "ACTIVE: Qin Shihuang Mode"
                if face_results.multi_face_landmarks:
                    face_landmarks = face_results.multi_face_landmarks[0]
                    forehead = face_landmarks.landmark[10]
                    chin = face_landmarks.landmark[152]
                    face_height = abs(forehead.y - chin.y) * h
                    if hat_img is not None:
                        hat_w = int(face_height * 1.8)
                        hat_h = int(hat_w * (hat_img.shape[0] / hat_img.shape[1]))
                        img = overlay_image(img, hat_img, int(forehead.x * w - hat_w / 2), int(forehead.y * h - hat_h * 0.85), size=(hat_w, hat_h))
                if bear_img is not None:
                    bear_w = int(w * 0.25)
                    bear_h = int(bear_w * (bear_img.shape[0] / bear_img.shape[1]))
                    img = overlay_image(img, bear_img, int(thumb_tip.x * w - bear_w / 2), int(thumb_tip.y * h - bear_h - 10), size=(bear_w, bear_h))

        cv2.putText(img, status_text, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        self.latest_filter = img.copy()
        return frame.from_ndarray(img, format="bgr24")

# --- 網頁畫面佈局 ---
ctx = webrtc_streamer(
    key="auto-meme-filter", 
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

if st.button("📸 Capture (拍照)", use_container_width=True):
    if ctx.video_processor and ctx.video_processor.latest_orig is not None:
        orig_rgb = cv2.cvtColor(ctx.video_processor.latest_orig, cv2.COLOR_BGR2RGB)
        filter_rgb = cv2.cvtColor(ctx.video_processor.latest_filter, cv2.COLOR_BGR2RGB)
        st.session_state.history.insert(0, (orig_rgb, filter_rgb))
        st.success("拍照成功！已加到下方紀錄中。")
    else:
        st.warning("請先點擊 START 開啟鏡頭再拍照喔！")

st.markdown("---")

if st.session_state.history:
    st.subheader("🖼️ 剛剛拍到的影像")
    current_orig, current_filter = st.session_state.history[0]
    
    col_orig, col_filt = st.columns(2)
    with col_orig:
        st.image(current_orig, caption="拍到的原影像", use_container_width=True)
    with col_filt:
        st.image(current_filter, caption="加上濾鏡後的影像", use_container_width=True)

    st.markdown("---")

    st.subheader("📜 歷史拍照紀錄")
    cols = st.columns(max(5, len(st.session_state.history)))
    for idx, (orig, filt) in enumerate(st.session_state.history):
        if idx < 5:
            with cols[idx]:
                st.image(filt, caption= f"紀錄 #{len(st.session_state.history)-idx}", use_container_width=True)
