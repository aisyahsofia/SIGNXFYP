import numpy as np
import cv2
import streamlit as st
from tensorflow import keras
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array

# Define the ASL emotion dictionary
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

def process_image(image):
    # Convert the image to grayscale
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize to 48x48, which is the expected input size for the model
    roi_gray = cv2.resize(img_gray, (48, 48), interpolation=cv2.INTER_AREA)

    if np.sum([roi_gray]) != 0:
        roi = roi_gray.astype('float') / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        prediction = classifier.predict(roi)[0]
        maxindex = int(np.argmax(prediction))
        finalout = emotion_dict[maxindex]
        return finalout
    return None

def main():
    # ASL detection application
    st.title("Real-Time ASL Sign Language Detection")
    activities = ["Home", "Phone Camera ASL Detection", "About"]
    choice = st.sidebar.selectbox("Select Activity", activities)

    if choice == "Home":
        st.markdown("""<div style="background-color:#6D7B8D;padding:10px">
                        <h4 style="color:white;text-align:center;">
                        ASL Sign Language Detection using Camera Feed</h4>
                        </div>""", unsafe_allow_html=True)
        st.write("""The application provides real-time detection of ASL signs through your phone's camera feed.""")

    elif choice == "Phone Camera ASL Detection":
        st.header("Phone Camera Live Feed for ASL")
        st.write("Click on the button below to use your phone's camera to detect ASL signs")

        # Phone camera input via Streamlit
        camera_image = st.camera_input("Take a photo", key="camera_input")

        if camera_image is not None:
            # Read the image from the uploaded file
            img = cv2.imdecode(np.frombuffer(camera_image.getvalue(), np.uint8), 1)

            # Process the image and display the result
            result = process_image(img)
            if result:
                st.write(f"Detected ASL sign: {result}")
            else:
                st.write("No sign detected or unable to recognize.")

    elif choice == "About":
        st.subheader("About this app")
        st.write("This app uses a trained model to detect ASL signs through your phone's camera feed.")

if __name__ == "__main__":
    main()
