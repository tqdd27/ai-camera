import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
import mediapipe as mp
import numpy as np

st.title("像個偉(偽)人一樣 📸")
st.write("👉 做出表情來變身！張開嘴巴：變身【愛因斯坦】")

if "history" not in st.session_state:
    st.session_state.history = []

@st.cache_data
def load_resources():
    # 確保圖片路徑正確
    e_hair = cv2.imread("assets/Einstein_hair.png", cv2.IMREAD_UNCHANGED)
    e_tongue = cv2.imread("assets/Einstein_tongue.png", cv2.IMREAD_UNCHANGED)
    return e_hair, e_tongue

einstein_hair, einstein_tongue = load_resources()

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

mp_face_mesh = mp.solutions.face_mesh

class VideoProcessor:
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        self.latest_orig = None
        self.latest_filter = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.latest_orig = img.copy()
        
        h, w, _ = img.shape
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_results = self.face_mesh.process(rgb_img)
        
        status_text = "Scanning... Make a gesture!"
        
        if face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0].landmark
            
            upper_lip = face_landmarks[13]  
            lower_lip = face_landmarks[14]  
            forehead = face_landmarks[10]   
            chin = face_landmarks[152]      
            
            face_height = abs(forehead.y - chin.y) * h
            lip_dist = abs(upper_lip.y - lower_lip.y) * h
            
            if lip_dist > (face_height * 0.15):
                status_text = "ACTIVE: Einstein Mode"
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                
                # ===== 愛因斯坦頭髮 =====
                if einstein_hair is not None:
                    left_face = face_landmarks[234]
                    right_face = face_landmarks[454]
                    face_width = abs(right_face.x - left_face.x) * w
                    hair_w = int(face_width * 1.75)
                    scale = einstein_hair.shape[0] / einstein_hair.shape[1]
                    hair_h = int(hair_w * scale)
                    hair_x = int(forehead.x * w - hair_w / 2.5)
                    hair_y = int(forehead.y * h - hair_h * 0.35)
                    img = overlay_image(img, einstein_hair, hair_x, hair_y, size=(hair_w, hair_h))
                
                # ===== 愛因斯坦舌頭 =====
                if einstein_tongue is not None:
                    left_mouth = face_landmarks[61]
                    right_mouth = face_landmarks[291]
                    mouth_width = abs(right_mouth.x - left_mouth.x) * w
                    tongue_w = int(mouth_width * 1.5)
                    scale = einstein_tongue.shape[0] / einstein_tongue.shape[1]
                    tongue_h = int(tongue_w * scale)
                    tongue_x = int(lower_lip.x * w - tongue_w / 2)
                    tongue_y = int(lower_lip.y * h + tongue_h * 0.4)
                    img = overlay_image(img, einstein_tongue, tongue_x, tongue_y, size=(tongue_w, tongue_h))
                    
        cv2.putText(img, status_text, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        self.latest_filter = img.copy()
        return frame.from_ndarray(img, format="bgr24")

    # 釋放 Mediapipe 資源
    def __del__(self):
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()

# --- 網頁畫面佈局 ---
ctx = webrtc_streamer(
    key="camera-step-by-step",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False}
)

if st.button("📸 Capture (拍照)", use_container_width=True):
    if ctx.video_processor and ctx.video_processor.latest_orig is not None:
        orig_rgb = cv2.cvtColor(ctx.video_processor.latest_orig, cv2.COLOR_BGR2RGB)
        filter_rgb = cv2.cvtColor(ctx.video_processor.latest_filter, cv2.COLOR_BGR2RGB)
        
        # 💡 【記憶體優化】：存入 history 前先將圖片縮小，避免爆記憶體
        max_width = 400
        h, w, _ = orig_rgb.shape
        if w > max_width:
            new_h = int(h * (max_width / w))
            orig_rgb = cv2.resize(orig_rgb, (max_width, new_h), interpolation=cv2.INTER_AREA)
            filter_rgb = cv2.resize(filter_rgb, (max_width, new_h), interpolation=cv2.INTER_AREA)
        
        st.session_state.history.insert(0, (orig_rgb, filter_rgb))
        if len(st.session_state.history) > 4:  # 💡 將上限從 8 張調降到 4 張更保險
            st.session_state.history.pop()
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
        st.image(current_filter, caption="濾鏡影像", use_container_width=True)

st.markdown("---")
st.subheader("📜 歷史拍照紀錄 (最多儲存 4 張)")
cols = st.columns(4)
for idx, (orig, filt) in enumerate(st.session_state.history):
    col_idx = idx % 4
    with cols[col_idx]:
        st.image(filt, use_container_width=True)
