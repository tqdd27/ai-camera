import os
# 【核心修復】強制讓 MediaPipe 在沒有螢幕的雲端 Linux 伺服器中順利執行
os.environ["XDG_RUNTIME_DIR"] = "/tmp/runtime-root"

import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2

import mediapipe as mp
import numpy as np
import time

# --- 新版 MediaPipe 導入方式 ---
import mediapipe.python.solutions.face_mesh as mp_face_mesh
import mediapipe.python.solutions.hands as mp_hands

# --- 設定 ---
st.set_page_config(page_title="AI 智慧修圖相機", page_icon="📸")
st.title("📸 AI 智慧修圖相機")
st.write("依據您的動作，自動套用不同的偉人濾鏡！")

# 初始化 session state 用於儲存照片
if "photo" not in st.session_state:
    st.session_state.photo = None

# --- 素材載入函數 ---
@st.cache_data
def load_resources():
    resources = {
        "hair": cv2.imread("hair.png", cv2.IMREAD_UNCHANGED),
        "tongue": cv2.imread("tongue.png", cv2.IMREAD_UNCHANGED),
        "confucius_hat": cv2.imread("confucius_hat.png", cv2.IMREAD_UNCHANGED),
        "beard": cv2.imread("beard.png", cv2.IMREAD_UNCHANGED),
        "sleeve": cv2.imread("sleeve.png", cv2.IMREAD_UNCHANGED),
        "emperor_hat": cv2.imread("emperor_hat.png", cv2.IMREAD_UNCHANGED),
        "bear": cv2.imread("bear.png", cv2.IMREAD_UNCHANGED),
        "holy_light": cv2.imread("holy_light.png", cv2.IMREAD_UNCHANGED),
        "tomato": cv2.imread("tomato.png", cv2.IMREAD_UNCHANGED),
    }
    return resources

# 載入素材
imgs = load_resources()

# --- 圖片覆蓋函數 (支援透明度) ---
def overlay_image(background, overlay, x, y, size=None):
    if overlay is None:
        return background
    
    bg_h, bg_w = background.shape[:2]
    if size is not None:
        overlay = cv2.resize(overlay, size, interpolation=cv2.INTER_AREA)
    h, w = overlay.shape[:2]
    if x >= bg_w or y >= bg_h or x + w <= 0 or y + h <= 0:
        return background
    
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

# --- 視訊處理類別 ---
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        self.hands = mp_hands.Hands(
            max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        self.current_filter = "無"
        self.processed_frame = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape
        
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_results = self.face_mesh.process(rgb_img)
        hand_results = self.hands.process(rgb_img)
        
        self.current_filter = "路易十六"
        output_img = img.copy()
        has_face = False
        face_height = 0
        forehead = (0, 0)
        nose_tip = (0, 0)
        chin = (0, 0)
        
        if face_results.multi_face_landmarks:
            has_face = True
            face_landmarks = face_results.multi_face_landmarks[0]
            
            def get_pt(idx):
                pt = face_landmarks.landmark[idx]
                return int(pt.x * w), int(pt.y * h)
            
            nose_tip = get_pt(4)
            forehead = get_pt(10)
            chin = get_pt(152)
            left_eye_top = get_pt(386)
            left_eye_bottom = get_pt(374)
            right_eye_top = get_pt(159)
            right_eye_bottom = get_pt(145)
            upper_lip = get_pt(13)
            lower_lip = get_pt(14)
            
            face_height = abs(chin[1] - forehead[1])
            eye_open_ratio = (abs(left_eye_top[1] - left_eye_bottom[1]) + abs(right_eye_top[1] - right_eye_bottom[1])) / (2 * face_height)
            mouth_open_ratio = abs(upper_lip[1] - lower_lip[1]) / face_height
            
            if eye_open_ratio < 0.015:
                self.current_filter = "釋迦牟尼佛"
                if imgs["holy_light"] is not None:
                    light_w = int(w * 0.8)
                    light_h = int(light_w * (imgs["holy_light"].shape[0] / imgs["holy_light"].shape[1]))
                    output_img = overlay_image(output_img, imgs["holy_light"], int(nose_tip[0] - light_w / 2), int(forehead[1] - light_h * 0.9), size=(light_w, light_h))
            elif mouth_open_ratio > 0.08:
                self.current_filter = "愛因斯坦"
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                output_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                if imgs["hair"] is not None:
                    hair_w = int(face_height * 1.5)
                    hair_h = int(hair_w * (imgs["hair"].shape[0] / imgs["hair"].shape[1]))
                    output_img = overlay_image(output_img, imgs["hair"], int(forehead[0] - hair_w / 2), int(forehead[1] - hair_h * 0.8), size=(hair_w, hair_h))
                if imgs["tongue"] is not None:
                    tongue_w = int(face_height * 0.4)
                    tongue_h = int(tongue_w * (imgs["tongue"].shape[0] / imgs["tongue"].shape[1]))
                    output_img = overlay_image(output_img, imgs["tongue"], int(lower_lip[0] - tongue_w / 2), int(lower_lip[1]), size=(tongue_w, tongue_h))

        if self.current_filter == "路易十六" and hand_results.multi_hand_landmarks:
            hand_landmarks = hand_results.multi_hand_landmarks[0]
            
            def get_hand_pt(idx):
                pt = hand_landmarks.landmark[idx]
                return int(pt.x * w), int(pt.y * h), pt.z
            
            wrist = get_hand_pt(0)
            thumb_tip = get_hand_pt(4)
            index_tip = get_hand_pt(8)
            middle_tip = get_hand_pt(12)
            ring_tip = get_hand_pt(16)
            pinky_tip = get_hand_pt(20)
            
            is_thumbs_up = thumb_tip[1] < index_tip[1] and thumb_tip[1] < middle_tip[1] and thumb_tip[1] < ring_tip[1] and thumb_tip[1] < pinky_tip[1]
            are_fingers_straight = index_tip[1] < wrist[1] and middle_tip[1] < wrist[1] and ring_tip[1] < wrist[1] and pinky_tip[1] < wrist[1]
            is_palm_facing_self = thumb_tip[2] > pinky_tip[2]
            
            if is_thumbs_up:
                self.current_filter = "秦始皇"
                if has_face:
                    if imgs["emperor_hat"] is not None:
                        hat_w = int(face_height * 1.8)
                        hat_h = int(hat_w * (imgs["emperor_hat"].shape[0] / imgs["emperor_hat"].shape[1]))
                        output_img = overlay_image(output_img, imgs["emperor_hat"], int(forehead[0] - hat_w / 2), int(forehead[1] - hat_h * 0.95), size=(hat_w, hat_h))
                if imgs["bear"] is not None:
                    bear_w = int(w * 0.4)
                    bear_h = int(bear_w * (imgs["bear"].shape[0] / imgs["bear"].shape[1]))
                    output_img = overlay_image(output_img, imgs["bear"], int(w - bear_w - 20), int(h - bear_h - 20), size=(bear_w, bear_h))
            elif are_fingers_straight and is_palm_facing_self:
                self.current_filter = "孔子"
                if has_face:
                    if imgs["confucius_hat"] is not None:
                        hat_w = int(face_height * 1.3)
                        hat_h = int(hat_w * (imgs["confucius_hat"].shape[0] / imgs["confucius_hat"].shape[1]))
                        output_img = overlay_image(output_img, imgs["confucius_hat"], int(forehead[0] - hat_w / 2), int(forehead[1] - hat_h * 0.8), size=(hat_w, hat_h))
                    if imgs["beard"] is not None:
                        beard_w = int(face_height * 0.8)
                        beard_h = int(beard_w * (imgs["beard"].shape[0] / imgs["beard"].shape[1]))
                        output_img = overlay_image(output_img, imgs["beard"], int(chin[0] - beard_w / 2), int(chin[1] - beard_h * 0.2), size=(beard_w, beard_h))
                if imgs["sleeve"] is not None:
                    sleeve_w = int(w * 0.3)
                    sleeve_h = int(sleeve_w * (imgs["sleeve"].shape[0] / imgs["sleeve"].shape[1]))
                    output_img = overlay_image(output_img, imgs["sleeve"], int(wrist[0] - sleeve_w / 2), int(wrist[1] - sleeve_h / 2), size=(sleeve_w, sleeve_h))

        if self.current_filter == "路易十六" and has_face:
            if imgs["tomato"] is not None:
                tomato_w = int(face_height * 1.4)
                tomato_h = int(tomato_w * (imgs["tomato"].shape[0] / imgs["tomato"].shape[1]))
                output_img = overlay_image(output_img, imgs["tomato"], int(nose_tip[0] - tomato_w / 2), int(nose_tip[1] - tomato_h / 2), size=(tomato_w, tomato_h))

        self.processed_frame = output_img
        cv2.putText(output_img, f"Filter: {self.current_filter}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return frame.from_ndarray(output_img, format="bgr24")

# --- Streamlit 網頁版面 ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("即時影像")
    webrtc_ctx = webrtc_streamer(
        key="meme-camera",
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

with col2:
    st.subheader("拍照控制")
    st.info("""
    **偉人解鎖密碼：**
    * 🤪 **愛因斯坦**：張開嘴巴（畫面變黑白、P頭髮+舌頭）
    * 📜 **孔子**：手心朝自己攤平（P帽子+鬍鬚+袖子）
    * 👑 **秦始皇**：比個讚 👍（P皇帝帽+北極熊）
    * 🪷 **釋迦牟尼佛**：閉上雙眼（頭頂散發聖光）
    * 🍅 **路易十六**：不做動作（預設頭變番茄）
    """)
    
    if st.button("📸 點我拍照", use_container_width=True):
        if webrtc_ctx.video_processor and webrtc_ctx.video_processor.processed_frame is not None:
            photo_bgr = webrtc_ctx.video_processor.processed_frame
            st.session_state.photo = cv2.cvtColor(photo_bgr, cv2.COLOR_BGR2RGB)
            st.success(f"成功捕捉！目前套用：{webrtc_ctx.video_processor.current_filter}")
        else:
            st.warning("請先允許瀏覽器開啟相機並點擊 Start 啟動。")

if st.session_state.photo is not None:
    st.markdown("---")
    st.subheader("🖼️ 您拍下的照片")
    st.image(st.session_state.photo, use_container_width=True)
    
    photo_bgr_dl = cv2.cvtColor(st.session_state.photo, cv2.COLOR_RGB2BGR)
    _, encoded = cv2.imencode('.jpg', photo_bgr_dl)
    
    st.download_button(
        label="📥 下載這張偉人照",
        data=encoded.tobytes(),
        file_name=f"great_man_{int(time.time())}.jpg",
        mime="image/jpeg",
        use_container_width=True
    )
