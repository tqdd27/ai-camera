import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import mediapipe as mp
import numpy as np
import time

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
    
    # 縮放覆蓋圖片
    if size is not None:
        overlay = cv2.resize(overlay, size, interpolation=cv2.INTER_AREA)
    
    h, w = overlay.shape[:2]
    
    # 檢查覆蓋區域是否在背景範圍內
    if x >= bg_w or y >= bg_h or x + w <= 0 or y + h <= 0:
        return background
    
    # 計算相交區域
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(bg_w, x + w), min(bg_h, y + h)
    
    overlay_x1, overlay_y1 = x1 - x, y1 - y
    overlay_x2, overlay_y2 = overlay_x1 + (x2 - x1), overlay_y1 + (y2 - y1)
    
    # 取得相交區域的切片
    crop_overlay = overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2]
    crop_bg = background[y1:y2, x1:x2]
    
    # 進行透明度混合
    if crop_overlay.shape[2] == 4:  # 如果有透明通道
        alpha = crop_overlay[:, :, 3] / 255.0
        alpha = np.expand_dims(alpha, axis=2)
        composite = crop_overlay[:, :, :3] * alpha + crop_bg * (1 - alpha)
        background[y1:y2, x1:x2] = composite
    else:  # 如果沒有透明通道，直接覆蓋
        background[y1:y2, x1:x2] = crop_overlay[:, :, :3]
        
    return background

# --- MediaPipe 初始化 ---
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

# --- 視訊處理類別 ---
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        # 初始化 MediaPipe 模型
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.hands = mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.current_filter = "無"
        self.processed_frame = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape
        
        # 轉換為 RGB 以供 MediaPipe 處理
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_results = self.face_mesh.process(rgb_img)
        hand_results = self.hands.process(rgb_img)
        
        # 預設狀態
        self.current_filter = "路易十六 (預設)"
        output_img = img.copy()
        
        # --- 偵測動作並套用濾鏡 ---
        
        # 1. 偵測臉部動作 (愛因斯坦, 孔子, 秦始皇, 釋迦牟尼佛, 路易十六)
        if face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0]
            
            # 取得關鍵點位置 (標準化座標 * 圖片寬高)
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
            
            # --- 判斷動作 ---
            
            # (a) 釋迦牟尼佛：閉眼 (眼睛高度小於門檻)
            eye_open_ratio = (abs(left_eye_top[1] - left_eye_bottom[1]) + abs(right_eye_top[1] - right_eye_bottom[1])) / (2 * face_height)
            if eye_open_ratio < 0.015:
                self.current_filter = "釋迦牟尼佛"
                # P 上聖光在頭上
                if imgs["holy_light"] is not None:
                    light_w = int(w * 0.8)
                    light_h = int(light_w * (imgs["holy_light"].shape[0] / imgs["holy_light"].shape[1]))
                    output_img = overlay_image(output_img, imgs["holy_light"], int(nose_tip[0] - light_w / 2), int(forehead[1] - light_h * 0.9), size=(light_w, light_h))
            
            # (b) 愛因斯坦：張開嘴巴 (嘴唇距離大於門檻)
            elif abs(upper_lip[1] - lower_lip[1]) > (face_height * 0.1):
                self.current_filter = "愛因斯坦"
                # 畫面轉黑白
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                output_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                # P 上頭髮、舌頭
                if imgs["hair"] is not None:
                    hair_w = int(face_height * 1.5)
                    hair_h = int(hair_w * (imgs["hair"].shape[0] / imgs["hair"].shape[1]))
                    output_img = overlay_image(output_img, imgs["hair"], int(forehead[0] - hair_w / 2), int(forehead[1] - hair_h * 0.8), size=(hair_w, hair_h))
                if imgs["tongue"] is not None:
                    tongue_w = int(face_height * 0.4)
                    tongue_h = int(tongue_w * (imgs["tongue"].shape[0] / imgs["tongue"].shape[1]))
                    output_img = overlay_image(output_img, imgs["tongue"], int(lower_lip[0] - tongue_w / 2), int(lower_lip[1]), size=(tongue_w, tongue_h))
            
            # (c) 路易十六：(預設動作) 頭 P 成番茄
            else:
                self.current_filter = "路易十六"
                if imgs["tomato"] is not None:
                    tomato_w = int(face_height * 1.3)
                    tomato_h = int(tomato_w * (imgs["tomato"].shape[0] / imgs["tomato"].shape[1]))
                    output_img = overlay_image(output_img, imgs["tomato"], int(nose_tip[0] - tomato_w / 2), int(nose_tip[1] - tomato_h / 2), size=(tomato_w, tomato_h))
                    
            # (d) 孔子 & 秦始皇 (需要手部動作，在此優先順序較低，若無人臉則不偵測手)
            if hand_results.multi_hand_landmarks and self.current_filter == "路易十六": # 僅在預設狀態下偵測手部動作
                hand_landmarks = hand_results.multi_hand_landmarks[0]
                
                # 取得手部關鍵點
                def get_hand_pt(idx):
                    pt = hand_landmarks.landmark[idx]
                    return int(pt.x * w), int(pt.y * h), pt.z # z 用於判斷深度
                
                wrist = get_hand_pt(0)
                thumb_tip = get_hand_pt(4)
                index_tip = get_hand_pt(8)
                middle_tip = get_hand_pt(12)
                ring_tip = get_hand_pt(16)
                pinky_tip = get_hand_pt(20)
                
                # --- 手部動作判斷 ---
                
                # (e) 秦始皇：比讚 (大拇指朝上，其他手指收起)
                is_thumbs_up = thumb_tip[1] < index_tip[1] and \
                               thumb_tip[1] < middle_tip[1] and \
                               thumb_tip[1] < ring_tip[1] and \
                               thumb_tip[1] < pinky_tip[1] and \
                               abs(thumb_tip[0] - wrist[0]) < abs(pinky_tip[0] - wrist[0]) # 簡單判斷大拇指在靠身體側
                
                if is_thumbs_up:
                    self.current_filter = "秦始皇"
                    # 重新使用原圖，因為路易十六已經 P 了番茄
                    output_img = img.copy() 
                    # P 上帽子、北極熊
                    if imgs["emperor_hat"] is not None:
                        hat_w = int(face_height * 1.8)
                        hat_h = int(hat_w * (imgs["emperor_hat"].shape[0] / imgs["emperor_hat"].shape[1]))
                        output_img = overlay_image(output_img, imgs["emperor_hat"], int(forehead[0] - hat_w / 2), int(forehead[1] - hat_h * 0.95), size=(hat_w, hat_h))
                    if imgs["bear"] is not None:
                        bear_w = int(w * 0.4)
                        bear_h = int(bear_w * (imgs["bear"].shape[0] / imgs["bear"].shape[1]))
                        output_img = overlay_image(output_img, imgs["bear"], int(w - bear_w - 20), int(h - bear_h - 20), size=(bear_w, bear_h))

                # (f) 孔子：手心朝向自己，手攤平 (所有手指伸直，大拇指 z 值較大)
                else:
                    # 簡單判斷手指是否伸直 (y 座標順序)
                    are_fingers_straight = index_tip[1] < wrist[1] and middle_tip[1] < wrist[1] and ring_tip[1] < wrist[1] and pinky_tip[1] < wrist[1]
                    # 簡單判斷手心朝向 (比較大拇指與小指的深度，需要更精確的場景可能需要手勢模型)
                    is_palm_facing_self = thumb_tip[2] > pinky_tip[2] # 預估大拇指較靠近相機

                    if are_fingers_straight and is_palm_facing_self:
                        self.current_filter = "孔子"
                        # 重新使用原圖
                        output_img = img.copy()
                        # P 上帽子、鬍鬚、袖子
                        if imgs["confucius_hat"] is not None:
                            hat_w = int(face_height * 1.3)
                            hat_h = int(hat_w * (imgs["confucius_hat"].shape[0] / imgs["confucius_hat"].shape[1]))
                            output_img = overlay_image(output_img, imgs["confucius_hat"], int(forehead[0] - hat_w / 2), int(forehead[1] - hat_h * 0.8), size=(hat_w, hat_h))
                        if imgs["beard"] is not None:
                            beard_w = int(face_height * 0.8)
                            beard_h = int(beard_w * (imgs["beard"].shape[0] / imgs["beard"].shape[1]))
                            output_img = overlay_image(output_img, imgs["beard"], int(chin[0] - beard_w / 2), int(chin[1] - beard_h * 0.2), size=(beard_w, beard_h))
                        if imgs["sleeve"] is not None:
                            # 簡單覆蓋在手腕位置
                            sleeve_w = int(w * 0.3)
                            sleeve_h = int(sleeve_w * (imgs["sleeve"].shape[0] / imgs["sleeve"].shape[1]))
                            output_img = overlay_image(output_img, imgs["sleeve"], int(wrist[0] - sleeve_w / 2), int(wrist[1] - sleeve_h / 2), size=(sleeve_w, sleeve_h))

        # 儲存處理後的畫面以便拍照
        self.processed_frame = output_img
        
        # 在畫面上顯示當前濾鏡名稱 (選用)
        cv2.putText(output_img, f"Filter: {self.current_filter}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return frame.from_ndarray(output_img, format="bgr24")

# --- Streamlit 畫面佈局 ---

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("即時影像")
    webrtc_ctx = webrtc_streamer(
        key="key",
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]} # 使用 Google 的 STUN 伺服器
    )

with col2:
    st.subheader("拍照控制")
    
    # 指導說明
    st.info("""
    **動作指南：**
    *   **愛因斯坦：** 張開嘴巴
    *   **孔子：** 手心朝自己，手攤平
    *   **秦始皇：** 比讚
    *   **釋迦牟尼佛：** 閉眼
    *   **路易十六：** (預設) 不做動作
    """)
    
    # 拍照按鈕
    if st.button("📸 拍照", use_container_width=True):
        if webrtc_ctx.video_processor and webrtc_ctx.video_processor.processed_frame is not None:
            # 取得處理後的最後一幀並轉換為 RGB 以便 Streamlit 顯示
            photo_bgr = webrtc_ctx.video_processor.processed_frame
            photo_rgb = cv2.cvtColor(photo_bgr, cv2.COLOR_BGR2RGB)
            st.session_state.photo = photo_rgb
            st.success(f"已使用「{webrtc_ctx.video_processor.current_filter}」濾鏡拍照！")
        else:
            st.warning("請先啟動相機。")

# --- 顯示照片 ---
if st.session_state.photo is not None:
    st.markdown("---")
    st.subheader("🖼️ 您的照片")
    st.image(st.session_state.photo, use_container_width=True)
    
    # 提供下載按鈕
    # 轉換為 BGR 再轉換為 JPG 位元組以便下載
    photo_bgr_download = cv2.cvtColor(st.session_state.photo, cv2.COLOR_RGB2BGR)
    _, img_encoded = cv2.imencode('.jpg', photo_bgr_download)
    img_bytes = img_encoded.tobytes()
    
    st.download_button(
        label="📥 下載照片",
        data=img_bytes,
        file_name=f"ai_photo_{int(time.time())}.jpg",
        mime="image/jpeg",
        use_container_width=True
    )
