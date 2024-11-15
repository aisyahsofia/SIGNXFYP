import numpy as np
import cv2
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Dictionary for ASL Alphabet (A-Y)
asl_dict = {
    'A': "A", 'B': "B", 'C': "C", 'D': "D", 'E': "E", 'F': "F", 'G': "G", 'H': "H", 'I': "I",
    'K': "K", 'L': "L", 'M': "M", 'N': "N", 'O': "O", 'P': "P", 'Q': "Q", 'R': "R", 'S': "S", 
    'T': "T", 'U': "U", 'V': "V", 'W': "W", 'X': "X", 'Y': "Y"
}

# Load your ASL model (ensure the model is correct and trained on these signs)
model = load_model('path_to_your_model.h5')  # Path to your trained model

# Load face detection model (for preprocessing purposes)
try:
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
except Exception:
    st.write("Error loading cascade classifiers")

class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Convert image to grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces (optional - depending on your model's needs)
        faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.3, minNeighbors=5)

        # For simplicity, we assume detecting the ASL sign in the entire image
        for (x, y, w, h) in faces:
            # You can crop around the detected face if your model is trained for that
            roi_gray = img_gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (64, 64), interpolation=cv2.INTER_AREA)

            # Preprocessing the image to feed it into the model
            if np.sum(roi_gray) != 0:
                roi = roi_gray.astype('float') / 255.0  # Normalize
                roi = img_to_array(roi)  # Convert to array
                roi = np.expand_dims(roi, axis=0)  # Add batch dimension

                # Predict the sign language letter (A-Y)
                prediction = model.predict(roi)
                maxindex = np.argmax(prediction)
                predicted_letter = list(asl_dict.keys())[maxindex]  # Get the ASL letter from dict

                # Draw the prediction on the image
                label_position = (x, y)
                cv2.putText(img, predicted_letter, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return img

def main():
    st.title("Real-Time ASL Detection Application")
    activities = ["Home", "Webcam ASL Detection", "About"]
    choice = st.sidebar.selectbox("Select Activity", activities)

    if choice == "Home":
        st.write("Welcome to the ASL Detection Application!")
        st.write("""
            This app detects American Sign Language (ASL) signs using your webcam.
            It can recognize signs from 'A' to 'Y' (without 'J' and 'Z').
        """)
    elif choice == "Webcam ASL Detection":
        st.header("Webcam Live Feed")
        st.write("Click on start to use webcam and detect your ASL sign.")
        webrtc_streamer(key="asl_sign_detection", video_transformer_factory=VideoTransformer)

    elif choice == "About":
        st.subheader("About this App")
        st.write("""
            This app is designed to detect and recognize American Sign Language (ASL) signs using a webcam feed. 
            It uses a pre-trained model for ASL letters from 'A' to 'Y' (excluding 'J' and 'Z').
        """)

if __name__ == "__main__":
    main()
