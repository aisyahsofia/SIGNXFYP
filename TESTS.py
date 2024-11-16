import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

def convert_keras_to_h5(model_path, output_path):
    try:
        # Load the model from the .keras file
        model = load_model(model_path)
        print(f"Model loaded from {model_path}")
        
        # Save the model to .h5 format
        model.save(output_path)
        print(f"Model saved as {output_path}")
    
    except Exception as e:
        print(f"Error during conversion: {e}")

if __name__ == "__main__":
    # Define paths for the input .keras model and output .h5 model
    model_path = r"C:\Users\puter\Downloads\final\data\keraspt1\AisyahSignX59.keras"  # Change this path if needed
    output_path = r"C:\Users\puter\Downloads\final\data\keraspt1\AisyahSignX59.h5"   # Define output path

    # Convert the model
    convert_keras_to_h5(model_path, output_path)

# Check the model's input shape to determine the expected input size
expected_input_size = model.input_shape[1]

# Label dictionary for mapping predicted indices to characters
label_dict = {
    '0': 'A', '1': 'B', '2': 'C', '3': 'D', '4': 'E',
    '5': 'F', '6': 'G', '7': 'H', '8': 'I', '9': 'K',
    '10': 'L', '11': 'M', '12': 'N', '13': 'O', '14': 'P',
    '15': 'Q', '16': 'R', '17': 'S', '18': 'T', '19': 'U',
    '20': 'V', '21': 'W', '22': 'X', '23': 'Y'
}

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def process_frame(frame, hands):
    """
    Process the input frame to detect hand gestures and predict sign language.
    """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Prepare data for prediction
            data_aux = []
            x_, y_ = [], []
            for landmark in hand_landmarks.landmark:
                x_.append(landmark.x)
                y_.append(landmark.y)
            min_x, min_y = min(x_), min(y_)

            for landmark in hand_landmarks.landmark:
                data_aux.append(landmark.x - min_x)
                data_aux.append(landmark.y - min_y)

            # Model prediction
            if len(data_aux) == expected_input_size:
                data_aux = np.asarray(data_aux).reshape(1, -1)
                prediction = model.predict(data_aux)
                predicted_class_index = np.argmax(prediction, axis=1)[0]
                predicted_probability = prediction[0][predicted_class_index]

                if predicted_probability >= 0.3:
                    predicted_character = label_dict.get(str(predicted_class_index), 'Unknown')
                else:
                    predicted_character = 'Unknown'

                # Display prediction on frame
                cv2.putText(
                    frame,
                    f'{predicted_character} ({predicted_probability * 100:.2f}%)',
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    2,
                    cv2.LINE_AA
                )
    return frame

# Streamlit UI
st.title("Real-Time Sign Language Recognition")
st.write("This app uses a webcam feed to recognize ASL gestures in real time.")

# Start button
if st.button("Start Webcam"):
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    stframe = st.empty()  # Placeholder for the video feed

    if not cap.isOpened():
        st.error("Could not open the webcam.")
    else:
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    st.warning("Failed to capture frame.")
                    break

                # Process frame
                frame = process_frame(frame, hands)

                # Convert to RGB for Streamlit display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                stframe.image(frame_rgb, channels="RGB")

                # Stop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cap.release()
else:
    st.info("Click 'Start Webcam' to begin.")

st.write("To exit the webcam feed, press 'q' in the window or stop the app.")
