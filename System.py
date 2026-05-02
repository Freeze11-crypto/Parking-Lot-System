import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import os
import gdown

st.set_page_config(page_title="Smart Parking", layout="centered")
st.title("🚗 Smart Parking Detection System")

MODEL_PATH = "best.pt"

# Download model if not present
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        url = "https://drive.google.com/file/d/1E2sRrRKfG5SwT2vnXW62H-c5MGq_8FP1/view?usp=drive_link"
        gdown.download(url, MODEL_PATH, quiet=False)
    return YOLO(MODEL_PATH)

model = load_model()

uploaded_file = st.file_uploader("Upload a parking lot image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    if st.button("Detect Parking Spaces"):
        with st.spinner('AI is analyzing...'):
            results = model.predict(image, conf=0.05)
            result_img = results[0].plot()
            result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            
            st.success("Detection Complete!")
            st.image(result_img, caption="Detected Parking Spaces", use_container_width=True)