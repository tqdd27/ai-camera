import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
import mediapipe as mp
import numpy as np

st.title("像個偉(偽)人一樣 📸")
st.write("👉 做出動作來變身：張嘴(愛因斯坦)、手攤平(孔子)、比讚(秦始皇)、閉眼(釋迦牟尼佛)、不做動作(路易十六)")

if "history" not in st.session_state:
    st.session_state.history = []

# --- 1. 載入所有名人的貼圖資源 (路徑修正：直接從根目錄讀取全小寫檔名) ---
@st.cache_data
def load_resources():
    resources = {
        "e_hair": cv2.imread("hair.png", cv2.IMREAD_UNCHANGED),
        "e_tongue": cv2.imread("tongue.png", cv2.IMREAD_UNCHANGED),
        "c_hat": cv2.imread("confucius_hat.png", cv2.IMREAD_UNCHANGED),
        "c_beard": cv2.imread("beard.png", cv2.IMREAD_UNCHANGED),
        "c_sleeve": cv2.imread("sleeve.png", cv2.IMREAD_UNCHANGED),
        "q_hat": cv2.imread("emperor_hat.png", cv2.IMREAD_UNCHANGED),  # 秦始皇帽子
        "q_bear": cv2.imread("bear.png", cv2.IMREAD_UNCHANGED),          # 北極熊
        "b_light": cv2.imread("holy_light.png", cv2.IMREAD_UNCHANGED),    # 聖光
        "l_tomato": cv2.imread("tomato.png", cv2.IMREAD_UNCHANGED)        # 番茄
    }
    return resources

res = load_resources()

# --- 2. 透明圖片疊加函式 ---
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
        background[y1:y2, x1:x2] = composite.astype(np.uint8)
    else:
        background[y1:y2, x1:x2] = crop_overlay[:, :, :3]
    return background

# --- 3. 視訊處理類別 (整合臉部網格與手部偵測) ---
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

class VideoProcessor:
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5
        )
        self.hands = mp_hands.Hands(
            max_num_hands=1, min_detection_confidence=0.5
        )
        self.latest_orig = None
        self.latest_filter = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # 鏡像翻轉，自拍比較直覺
        img = cv2.flip(img, 1)
        self.latest_orig = img.copy()
        
        h, w, _ = img.shape
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 同步執行 AI 偵測
        face_results = self.face_mesh.process(rgb_img)
        hand_results = self.hands.process(rgb_img)
        
        # 核心狀態機：預設為路易十六（不做動作）
        role = "louis"
        status_text = "ACTIVE: Louis XVI Mode"
        
        # --- 動作判斷邏輯 ---
        # A. 檢查臉部特徵 (愛因斯坦、釋迦牟尼佛)
        if face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0].landmark
            upper_lip = face_landmarks[13]
            lower_lip = face_landmarks[14]
            forehead = face_landmarks[10]
            chin = face_landmarks[152]
            
            # 偵測閉眼特徵點
            left_eye_top = face_landmarks[159]
            left_eye_bottom = face_landmarks[145]
            right_eye_top = face_landmarks[386]
            right_eye_bottom = face_landmarks[374]
            
            face_height = abs(forehead.y - chin.y) * h
            lip_dist = abs(upper_lip.y - lower_lip.y) * h
            eye_dist = ((abs(left_eye_top.y - left_eye_bottom.y) + abs(right_eye_top.y - right_eye_bottom.y)) / 2) * h

            if lip_dist > (face_height * 0.15):
                role = "einstein"
                status_text = "ACTIVE: Einstein Mode"
            elif eye_dist < (face_height * 0.025):  # 閉眼閥值
                role = "buddha"
                status_text = "ACTIVE: Buddha Mode"

        # B. 檢查手部特徵 (秦始皇、孔子)
        if hand_results and hand_results.multi_hand_landmarks:
            hand_landmarks = hand_results.multi_hand_landmarks[0].landmark
            thumb_tip = hand_landmarks[4]
            thumb_ip = hand_landmarks[3]
            index_mcp = hand_landmarks[5]
            index_tip = hand_landmarks[8]
            middle_tip = hand_landmarks[12]
            ring_tip = hand_landmarks[16]
            
            # 秦始皇：比讚 (大拇指朝上，食指中指收起)
            is_thumb_up = (thumb_tip.y < thumb_ip.y) and (index_tip.y > index_mcp.y) and (middle_tip.y > index_mcp.y)
            # 孔子：手攤平 (手指皆伸直高於主要關節)
            is_flat_hand = (index_tip.y < index_mcp.y) and (middle_tip.y < hand_landmarks[9].y) and (ring_tip.y < hand_landmarks[13].y)
            
            if is_thumb_up:
                role = "qin"
                status_text = "ACTIVE: Qin Shihuang Mode"
            elif is_flat_hand:
                role = "confucius"
                status_text = "ACTIVE: Confucius Mode"

        # --- 濾鏡與貼圖繪製邏輯 ---
        if face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0].landmark
            forehead = face_landmarks[10]
            chin = face_landmarks[152]
            nose = face_landmarks[1]
            left_face = face_landmarks[234]
            right_face = face_landmarks[454]
            
            face_width = abs(right_face.x - left_face.x) * w
            face_height = abs(forehead.y - chin.y) * h

            # 1. 愛因斯坦：轉黑白 + 頭髮 + 舌頭
            if role == "einstein":
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                if res["e_hair"] is not None:
                    hair_w = int(face_width * 1.75)
                    hair_h = int(hair_w * (res["e_hair"].shape[0] / res["e_hair"].shape[1]))
                    img = overlay_image(img, res["e_hair"], int(forehead.x * w - hair_w / 2.5), int(forehead.y * h - hair_h * 0.35), size=(hair_w, hair_h))
                if res["e_tongue"] is not None:
                    mouth_width = abs(face_landmarks[291].x - face_landmarks[61].x) * w
                    tongue_w = int(mouth_width * 1.5)
                    tongue_h = int(tongue_w * (res["e_tongue"].shape[0] / res["e_tongue"].shape[1]))
                    img = overlay_image(img, res["e_tongue"], int(face_landmarks[14].x * w - tongue_w / 2), int(face_landmarks[14].y * h + tongue_h * 0.4), size=(tongue_w, tongue_h))

            # 2. 孔子：帽子 + 鬍鬚 + 袖子
            elif role == "confucius":
                if res["c_hat"] is not None:
                    hat_w = int(face_width * 1.5)
                    hat_h = int(hat_w * (res["c_hat"].shape[0] / res["c_hat"].shape[1]))
                    img = overlay_image(img, res["c_hat"], int(forehead.x * w - hat_w / 2), int(forehead.y * h - hat_h * 0.8), size=(hat_w, hat_h))
                if res["c_beard"] is not None:
                    beard_w = int(face_width * 1.0)
                    beard_h = int(beard_w * (res["c_beard"].shape[0] / res["c_beard"].shape[1]))
                    img = overlay_image(img, res["c_beard"], int(chin.x * w - beard_w / 2), int(chin.y * h - beard_h * 0.1), size=(beard_w, beard_h))
                if res["c_sleeve"] is not None:
                    sleeve_w = int(w * 0.28)
                    sleeve_h = int(sleeve_w * (res["c_sleeve"].shape[0] / res["c_sleeve"].shape[1]))
                    if hand_results and hand_results.multi_hand_landmarks:
                        hx = int(hand_landmarks[9].x * w - sleeve_w / 2)
                        hy = int(hand_landmarks[9].y * h - sleeve_h / 2)
                        img = overlay_image(img, res["c_sleeve"], hx, hy, size=(sleeve_w, sleeve_h))
                    else:
                        img = overlay_image(img, res["c_sleeve"], w - sleeve_w - 20, h - sleeve_h - 20, size=(sleeve_w, sleeve_h))

            # 3. 秦始皇：帽子 + 北極熊
            elif role == "qin":
                if res["q_hat"] is not None:
                    hat_w = int(face_width * 1.8)
                    hat_h = int(hat_w * (res["q_hat"].shape[0] / res["q_hat"].shape[1]))
                    img = overlay_image(img, res["q_hat"], int(forehead.x * w - hat_w / 2), int(forehead.y * h - hat_h * 0.85), size=(hat_w, hat_h))
                if res["q_bear"] is not None:
                    bear_w = int(w * 0.25)
                    bear_h = int(bear_w * (res["q_bear"].shape[0] / res["q_bear"].shape[1]))
                    img = overlay_image(img, res["q_bear"], 20, h - bear_h - 20, size=(bear_w, bear_h))

            # 4. 釋迦牟尼佛：閉眼頭頂聖光
            elif role == "buddha":
                if res["b_light"] is not None:
                    light_w = int(face_height * 2.2)
                    light_h = int(light_w * (res["b_light"].shape[0] / res["b_light"].shape[1]))
                    img = overlay_image(img, res["b_light"], int(forehead.x * w - light_w / 2), int(forehead.y * h - light_h * 0.75), size=(light_w, light_h))

            # 5. 路易十六：預設番茄蓋住整張臉
            elif role == "louis":
                if res["l_tomato"] is not None:
                    tomato_w = int(face_width * 1.5)
                    tomato_h = int(tomato_w * (res["l_tomato"].shape[0] / res["l_tomato"].shape[1]))
                    img = overlay_image(img, res["l_tomato"], int(nose.x * w - tomato_w / 2), int(nose.y * h - tomato_h / 2), size=(tomato_w, tomato_h))

        # 顯示變身提示文字
        cv2.putText(img, status_text, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        self.latest_filter = img.copy()
        return frame.from_ndarray(img, format="bgr24")

    def __del__(self):
        if hasattr(self, 'face_mesh'): self.face_mesh.close()
        if hasattr(self, 'hands'): self.hands.close()

# --- 4. 網頁串流畫面與拍照功能 ---
ctx = webrtc_streamer(
    key="camera-all-heroes", 
    video_processor_factory=VideoProcessor, 
    media_stream_constraints={"video": True, "audio": False}
)

if st.button("📸 Capture (拍照)", use_container_width=True):
    if ctx.video_processor and ctx.video_processor.latest_orig is not None:
        orig_rgb = cv2.cvtColor(ctx.video_processor.latest_orig, cv2.COLOR_BGR2RGB)
        filter_rgb = cv2.cvtColor(ctx.video_processor.latest_filter, cv2.COLOR_BGR2RGB)
        
        # 記憶體縮圖優化
        max_width = 400
        h, w, _ = orig_rgb.shape
        if w > max_width:
            new_h = int(h * (max_width / w))
            orig_rgb = cv2.resize(orig_rgb, (max_width, new_h), interpolation=cv2.INTER_AREA)
            filter_rgb = cv2.resize(filter_rgb, (max_width, new_h), interpolation=cv2.INTER_AREA)
            
        st.session_state.history.insert(0, (orig_rgb, filter_rgb))
        if len(st.session_state.history) > 4:
            st.session_state.history.pop()
        st.success("拍照成功！已加到下方紀錄中。")
    else:
        st.warning("請先點擊 START 開啟鏡頭再拍照喔！")

st.markdown("---")
if st.session_state.history:
    st.subheader("🖼️ 剛剛拍到的影像")
    current_orig, current_filter = st.session_state.history[0]
    col_orig, col_filt = st.columns(2)
    with col_orig: st.image(current_orig, caption="拍到的原影像", use_container_width=True)
    with col_filt: st.image(current_filter, caption="濾鏡影像", use_container_width=True)

    st.markdown("---")
    st.subheader("📜 歷史拍照紀錄 (最多儲存 4 張)")
    cols = st.columns(4)
    for idx, (orig, filt) in enumerate(st.session_state.history):
        with cols[idx % 4]: st.image(filt, use_container_width=True)
