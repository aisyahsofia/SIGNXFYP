import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image

# Load the Keras model
model = load_model('./models/AisyahSignX59.keras')

# Check the model's input shape to determine the expected input size
expected_input_size = model.input_shape[1]  # Adjust based on your model's input shape

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,  # Set to False for continuous detection
    max_num_hands=1,  # Detect one hand at a time for simplicity
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Label dictionary for mapping predicted indices to characters
label_dict = {
    '0': 'A', '1': 'B', '2': 'C', '3': 'D', '4': 'E', '5': 'F', '6': 'G',
    '7': 'H', '8': 'I', '9': 'K', '10': 'L', '11': 'M', '12': 'N', '13': 'O',
    '14': 'P', '15': 'Q', '16': 'R', '17': 'S', '18': 'T', '19': 'U', '20': 'V',
    '21': 'W', '22': 'X', '23': 'Y'
}

# Streamlit UI setup
st.title("Sign Language Recognition")
st.write("Use your webcam to recognize sign language gestures.")

# Camera input
camera_input = st.camera_input("Take a photo")

# Check if a frame is captured
if camera_input:
    # Convert the uploaded image to a frame
    frame = np.array(camera_input)
    
    # Convert frame to RGB for MediaPipe processing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # If hand landmarks are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Extract normalized landmark coordinates
            data_aux = []
            x_ = []
            y_ = []
            for landmark in hand_landmarks.landmark:
                x_.append(landmark.x)
                y_.append(landmark.y)

            # Create feature vector
            min_x, min_y = min(x_), min(y_)
            for landmark in hand_landmarks.landmark:
                data_aux.append(landmark.x - min_x)
                data_aux.append(landmark.y - min_y)

            # Ensure correct data format for model prediction
            if len(data_aux) == expected_input_size:
                data_aux = np.asarray(data_aux).reshape(1, -1)
                prediction = model.predict(data_aux)

                # Get predicted class and probability
                predicted_class_index = np.argmax(prediction, axis=1)[0]
                predicted_probability = prediction[0][predicted_class_index]

                if predicted_probability >= 0.3:
                    predicted_character = label_dict.get(str(predicted_class_index), 'Unknown')
                else:
                    predicted_character = 'Unknown'

                # Draw prediction on the frame
                x1 = int(min(x_) * frame.shape[1]) - 10
                y1 = int(min(y_) * frame.shape[0]) - 10
                x2 = int(max(x_) * frame.shape[1]) + 10
                y2 = int(max(y_) * frame.shape[0]) + 10

                cv2.rectangle(frame, (x1, y1), (x2, y2), (72, 61, 139), 4)
                cv2.putText(frame, f'{predicted_character} ({predicted_probability * 100:.2f}%)',
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (72, 61, 139), 3, cv2.LINE_AA)

    # Convert the frame to an image for Streamlit
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    image = Image.fromarray(frame_bgr)

    # Display the image with the prediction
    st.image(image, caption="Sign Language Recognition", use_column_width=True)
