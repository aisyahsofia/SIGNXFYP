import numpy as np
import cv2
import streamlit as st
from tensorflow import keras
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# load model
emotion_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 
                9: 'K', 10: 'L', 11: 'M', 12: 'N', 13: 'O', 14: 'P', 15: 'Q', 16: 'R', 17: 'S', 
                18: 'T', 19: 'U', 20: 'V', 21: 'W', 22: 'X', 23: 'Y'}

# Load model configuration
json_file = open('path_to_your_local_model/compile.json', 'r')  # Update path here
loaded_model_json = json_file.read()
json_file.close()

# Load model architecture
classifier = model_from_json(loaded_model_json)

# Load model weights
classifier.load_weights('path_to_your_local_model/keraspt1/asl_model.h5')  # Update path here

# Load face detection (optional)
try:
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
except Exception:
    st.write("Error loading cascade classifiers")

class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Image gray
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(image=img_gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img=img, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=2)
            roi_gray = img_gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                prediction = classifier.predict(roi)[0]
                maxindex = int(np.argmax(prediction))
                finalout = emotion_dict[maxindex]
                output = str(finalout)
            label_position = (x, y)
            cv2.putText(img, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return img

def main():
    # ASL detection application #
    st.title("Real-Time ASL Sign Language Detection")
    activities = ["Home", "Webcam ASL Detection", "About"]
    choice = st.sidebar.selectbox("Select Activity", activities)

    if choice == "Home":
        st.markdown("""<div style="background-color:#6D7B8D;padding:10px">
                        <h4 style="color:white;text-align:center;">
                        ASL Sign Language Detection using Webcam Feed</h4>
                        </div>""", unsafe_allow_html=True)
        st.write("""
                The application provides real-time detection of ASL signs through webcam feed.
                """)
    elif choice == "Webcam ASL Detection":
        st.header("Webcam Live Feed for ASL")
        st.write("Click on start to use webcam and detect ASL signs")
        webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)

    elif choice == "About":
        st.subheader("About this app")
        st.write("This app uses a trained model to detect ASL signs through webcam feed.")

if __name__ == "__main__":
    main()
