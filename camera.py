import streamlit as st
import cv2

st.title("Camera Test in Streamlit")

# Checkbox to open/close camera
run_camera = st.checkbox("Open Camera")

FRAME_WINDOW = st.image([])

if run_camera:
    cap = cv2.VideoCapture(1)
    if cap.isOpened():
        while run_camera:
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                FRAME_WINDOW.image(frame_rgb)
            else:
                st.warning("Camera frame not available.")
                break
    cap.release()
