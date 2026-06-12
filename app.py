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
st.title("歷史名人 AI 濾鏡相機 📸")
st.write("👉 根據動作自動變換角色：張嘴(愛因斯坦)、手攤平(孔子)、比讚(秦始皇)、閉眼(釋迦牟尼佛)、不做動作(路易十六)")

if "history" not in st.session_state:
    st.session_state.history = []

# 擴充資源載入，確保所有貼圖都進來
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
        "tomato": cv2.imread("tomato.png", cv2.IMREAD_UNCHANGED)
    }
    return resources

res = load_resources()

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

class VideoProcessor:
    def __init__(self):
        # 為了偵測閉眼，refine_landmarks 設為 True 可以取得更精細的眼睛網格
        self.face_mesh = FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)
        self.hands = Hands(max_num_hands=1, min_detection_confidence=0.5)
        self.latest_orig = None
        self.latest_filter = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # 鏡像翻轉（自拍比較直覺）
        img = cv2.flip(img, 1)
        self.latest_orig = img.copy()
        
        h, w, _ = img.shape
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        face_results = self.face_mesh.process(rgb_img)
        hand_results = self.hands.process(rgb_img)
        
        # 核心狀態機：預設為路易十六
        status_text = "ACTIVE: Louis XVI Mode"
        role = "louis"
        
        # --- 1. 動作偵測階段 ---
        # 檢查臉部動作
        if face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0]
            
            # 【愛因斯坦】張嘴偵測
            upper_lip = face_landmarks.landmark[13]
            lower_lip = face_landmarks.landmark[14]
            lip_dist = abs(upper_lip.y - lower_lip.y) * h
            
            # 【釋迦牟尼佛】閉眼偵測 (計算上下眼瞼距離)
            left_eye_top = face_landmarks.landmark[159]
            left_eye_bottom = face_landmarks.landmark[145]
            right_eye_top = face_landmarks.landmark[386]
            right_eye_bottom = face_landmarks.landmark[374]
            eye_dist = ((abs(left_eye_top.y - left_eye_bottom.y) + abs(right_eye_top.y - right_eye_bottom.y)) / 2) * h

            # 臉部參考高度
            forehead = face_landmarks.landmark[10]
            chin = face_landmarks.landmark[152]
            face_height = abs(forehead.y - chin.y) * h
            
            if lip_dist > (face_height * 0.15):
                role = "einstein"
                status_text = "ACTIVE: Einstein Mode"
            elif eye_dist < (face_height * 0.025):  # 閉眼閥值
                role = "buddha"
                status_text = "ACTIVE: Buddha Mode"

        # 檢查手部動作（若臉部已有特殊動作，手勢可選擇不覆蓋或覆蓋，此處設定手勢優先度高）
        if hand_results and hand_results.multi_hand_landmarks:
            hand_landmarks = hand_results.multi_hand_landmarks[0]
            
            thumb_tip = hand_landmarks.landmark[4]
            thumb_ip = hand_landmarks.landmark[3]
            index_mcp = hand_landmarks.landmark[5]
            index_tip = hand_landmarks.landmark[8]
            middle_tip = hand_landmarks.landmark[12]
            ring_tip = hand_landmarks.landmark[16]
            pinky_tip = hand_landmarks.landmark[20]
            
            # 【秦始皇】比讚：大拇指高於指節，其餘手指收起（低於指節）
            is_thumb_up = (thumb_tip.y < thumb_ip.y) and (index_tip.y > index_mcp.y) and (middle_tip.y > index_mcp.y)
            
            # 【孔子】手攤平：指尖皆高於各自的 MCP 關節
            is_flat_hand = (index_tip.y < index_mcp.y) and (middle_tip.y < hand_landmarks.landmark[9].y) and (ring_tip.y < hand_landmarks.landmark[13].y)
            
            if is_thumb_up:
                role = "qin"
                status_text = "ACTIVE: Qin Shihuang Mode"
            elif is_flat_hand:
                role = "confucius"
                status_text = "ACTIVE: Confucius Mode"

        # --- 2. 特效繪製階段 ---
        if face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0]
            forehead = face_landmarks.landmark[10]
            chin = face_landmarks.landmark[152]
            nose = face_landmarks.landmark[1]
            face_height = abs(forehead.y - chin.y) * h
            face_width = face_height  # 概算寬度
            
            # 根據不同角色渲染
            if role == "einstein":
                # 畫面轉黑白
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                # P 上頭髮
                hair_w = int(face_height * 1.6)
                hair_h = int(hair_w * (res["hair"].shape[0] / res["hair"].shape[1])) if res["hair"] is not None else 10
                img = overlay_image(img, res["hair"], int(forehead.x * w - hair_w / 2), int(forehead.y * h - hair_h * 0.8), size=(hair_w, hair_h))
                # P 上舌頭
                tongue_w = int(face_width * 0.4)
                tongue_h = int(tongue_w * (res["tongue"].shape[0] / res["tongue"].shape[1])) if res["tongue"] is not None else 10
                img = overlay_image(img, res["tongue"], int(face_landmarks.landmark[14].x * w - tongue_w / 2), int(face_landmarks.landmark[14].y * h), size=(tongue_w, tongue_h))
                
            elif role == "confucius":
                # P 上帽子
                hat_w = int(face_height * 1.5)
                hat_h = int(hat_w * (res["confucius_hat"].shape[0] / res["confucius_hat"].shape[1])) if res["confucius_hat"] is not None else 10
                img = overlay_image(img, res["confucius_hat"], int(forehead.x * w - hat_w / 2), int(forehead.y * h - hat_h * 0.85), size=(hat_w, hat_h))
                # P 上鬍鬚
                beard_w = int(face_width * 0.8)
                beard_h = int(beard_w * (res["beard"].shape[0] / res["beard"].shape[1])) if res["beard"] is not None else 10
                img = overlay_image(img, res["beard"], int(chin.x * w - beard_w / 2), int(chin.y * h - beard_h * 0.2), size=(beard_w, beard_h))
                # P 上袖子 (若有偵測到手就跟隨手，沒有就放右下角)
                sleeve_w = int(w * 0.3)
                sleeve_h = int(sleeve_w * (res["sleeve"].shape[0] / res["sleeve"].shape[1])) if res["sleeve"] is not None else 10
                if hand_results and hand_results.multi_hand_landmarks:
                    hand_center_x = hand_landmarks.landmark[9].x * w
                    hand_center_y = hand_landmarks.landmark[9].y * h
                    img = overlay_image(img, res["sleeve"], int(hand_center_x - sleeve_w / 2), int(hand_center_y - sleeve_h * 0.3), size=(sleeve_w, sleeve_h))
                else:
                    img = overlay_image(img, res["sleeve"], w - sleeve_w - 20, h - sleeve_h - 20, size=(sleeve_w, sleeve_h))
                    
            elif role == "qin":
                # P 上皇帝帽
                hat_w = int(face_height * 1.8)
                hat_h = int(hat_w * (res["emperor_hat"].shape[0] / res["emperor_hat"].shape[1])) if res["emperor_hat"] is not None else 10
                img = overlay_image(img, res["emperor_hat"], int(forehead.x * w - hat_w / 2), int(forehead.y * h - hat_h * 0.85), size=(hat_w, hat_h))
                # P 上北極熊 (放在畫面的左下角固定位置)
                bear_w = int(w * 0.25)
                bear_h = int(bear_w * (res["bear"].shape[0] / res["bear"].shape[1])) if res["bear"] is not None else 10
                img = overlay_image(img, res["bear"], 20, h - bear_h - 20, size=(bear_w, bear_h))
                
            elif role == "buddha":
                # P 上聖光在頭上
                light_w = int(face_height * 2.2)
                light_h = int(light_w * (res["holy_light"].shape[0] / res["holy_light"].shape[1])) if res["holy_light"] is not None else 10
                img = overlay_image(img, res["holy_light"], int(forehead.x * w - light_w / 2), int(forehead.y * h - light_h * 0.75), size=(light_w, light_h))
                
            elif role == "louis":
                # 不做動作，直接把頭 P 成番茄（完美蓋住臉）
                tomato_w = int(face_width * 1.4)
                tomato_h = int(tomato_w * (res["tomato"].shape[0] / res["tomato"].shape[1])) if res["tomato"] is not None else 10
                img = overlay_image(img, res["tomato"], int(nose.x * w - tomato_w / 2), int(nose.y * h - tomato_h / 2), size=(tomato_w, tomato_h))

        # 輸出狀態文字
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
                st.image(filt, caption=f"紀錄 #{len(st.session_state.history)-idx}", use_container_width=True)
