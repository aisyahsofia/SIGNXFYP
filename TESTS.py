import streamlit as st
import pandas as pd
import hashlib
import random
import cv2
import numpy as np
import os
import mediapipe as mp
from tensorflow.keras.models import load_model
from PIL import Image

# Load the Keras model (adjust the path as needed)
model = load_model(r"C:\xampp\htdocs\Project\GitSIgn\Compile\data\keras\compile.keras")

# Initialize video capture
cap = cv2.VideoCapture(0)

# Set up MediaPipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)

# Define the ASL alphabet mapping (A to Y, excluding J and Z)
asl_alphabet = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I',
    9: 'K', 10: 'L', 11: 'M', 12: 'N', 13: 'O', 14: 'P', 15: 'Q', 16: 'R',
    17: 'S', 18: 'T', 19: 'U', 20: 'V', 21: 'W', 22: 'X', 23: 'Y'
}

# Function to predict ASL from webcam input
def predict_asl():
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        return None, None  # Exit if the frame is not captured

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        # Prepare data for prediction
        data_aux = np.asarray(data_aux).reshape(1, -1)  # Reshape for the model input
        prediction = model.predict(data_aux)

        # Get the predicted class and its confidence
        predicted_class_index = np.argmax(prediction, axis=1)[0]  # Index of the predicted class (e.g., 0 to 23)
        predicted_probability = prediction[0][predicted_class_index]  # Probability of the predicted class

        # Check if the probability is above the threshold (30%)
        if predicted_probability >= 0.3:
            predicted_character = asl_alphabet.get(predicted_class_index, 'Unknown')
        else:
            predicted_character = 'Unknown'  # Fallback for low confidence

        # Draw the prediction on the frame
        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        cv2.rectangle(frame, (x1, y1), (x2, y2), (75, 75, 75), 4)
        cv2.putText(frame, f'{predicted_character} ({predicted_probability * 100:.2f}%)', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (75, 75, 75), 3, cv2.LINE_AA)

    # Convert the frame to RGB and display it in Streamlit
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    return img, predicted_character


# File paths
USERS_FILE = "users.csv"
PROGRESS_FILE = "progress.csv"
SIGN_DATA_FILE = "sign_language_data.csv"

# Base URL for GitHub raw files
BASE_URL = "https://raw.githubusercontent.com/aisyahsofia/SIGNXFYP/main/"

# Sign language data for training
SIGN_LANGUAGE_DATA = {
    "Hello": f"{BASE_URL}HELLO%20ASL.mp4",
    "Thank You": f"{BASE_URL}THANKYOU.mp4",
    "Sorry": f"{BASE_URL}SORRY%20ASL.mp4",
    "Please": f"{BASE_URL}PLEASE%20ASL.mp4",
    "Yes": f"{BASE_URL}YES%20ASL.mp4",
    "No": f"{BASE_URL}NO%20ASL.mp4",
    "How Are You?": f"{BASE_URL}HOWAREYOU%20ASL.mp4",
    "My Name Is...": f"{BASE_URL}MYNAMEIS%20ASL.mp4",
    "What Is Your Name?": f"{BASE_URL}WHATISYOURNAME%20ASL.mp4",
    "I Am Deaf": f"{BASE_URL}IMDEAF%20ASL.mp4",
    "I Am Hearing": f"{BASE_URL}IMHEARING%20ASL.mp4",
    "Where Is the Toilet?": f"{BASE_URL}WHEREISTHETOILET%20ASL.mp4",
    "Help me": f"{BASE_URL}HELPME%20ASL.mp4",
    "I Love You": f"{BASE_URL}ILOVEYOU%20ASL.mp4",
    "See You Later": f"{BASE_URL}SEEYOULATER%20ASL.mp4",
    "Good Morning": f"{BASE_URL}GOODMORNING%20ASL.mp4",
    "Good Afternoon": f"{BASE_URL}GOODAFTERNOON%20ASL.mp4",
    "Good Evening": f"{BASE_URL}GOODEVENING%20ASL.mp4",
    "Good Night": f"{BASE_URL}GOODNIGHT%20ASL.mp4",
    "Goodbye": f"{BASE_URL}GOODBYE%20ASL.mp4",
}

# Basic ASL alphabet
ASL_ALPHABET = {
    'A': f"{BASE_URL}A%20ASL.mp4",
    'B': f"{BASE_URL}B%20ASL.mp4",
    'C': f"{BASE_URL}C%20ASL.mp4",
    'D': f"{BASE_URL}D%20ASL.mp4",
    'E': f"{BASE_URL}E%20ASL.mp4",
    'F': f"{BASE_URL}F%20ASL.mp4",
    'G': f"{BASE_URL}G%20ASL.mp4",
    'H': f"{BASE_URL}H%20ASL.mp4",
    'I': f"{BASE_URL}I%20ASL.mp4",
    'J': f"{BASE_URL}J%20ASL.mp4",
    'K': f"{BASE_URL}K%20ASL.mp4",
    'L': f"{BASE_URL}L%20ASL.mp4",
    'M': f"{BASE_URL}M%20ASL.mp4",
    'N': f"{BASE_URL}N%20ASL.mp4",
    'O': f"{BASE_URL}O%20ASL.mp4",
    'P': f"{BASE_URL}P%20ASL.mp4",
    'Q': f"{BASE_URL}Q%20ASL.mp4",
    'R': f"{BASE_URL}R%20ASL.mp4",
    'S': f"{BASE_URL}S%20ASL.mp4",
    'T': f"{BASE_URL}T%20ASL.mp4",
    'U': f"{BASE_URL}U%20ASL.mp4",
    'V': f"{BASE_URL}V%20ASL.mp4",
    'W': f"{BASE_URL}W%20ASL.mp4",
    'X': f"{BASE_URL}X%20ASL.mp4",
    'Y': f"{BASE_URL}Y%20ASL.mp4",
    'Z': f"{BASE_URL}Z%20ASL.mp4"
}

# Hashing function for passwords
def hash_password(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

# Save user data to a CSV
def save_user_data(users_data):
    users_data.to_csv(USERS_FILE, index=False)

# Load user data from a CSV
def load_user_data():
    try:
        return pd.read_csv(USERS_FILE)
    except FileNotFoundError:
        return pd.DataFrame(columns=["username", "password"])

# Save progress data
def save_progress_data(progress_data):
    progress_data.to_csv(PROGRESS_FILE, index=False)

# Load progress data
def load_progress_data():
    try:
        return pd.read_csv(PROGRESS_FILE)
    except FileNotFoundError:
        return pd.DataFrame(columns=["username", "sign", "attempts", "correct"])

# User authentication function
def authenticate_user(username, password):
    users_data = load_user_data()
    user_data = users_data[users_data["username"] == username]
    if not user_data.empty:
        stored_hash = user_data.iloc[0]["password"]
        if stored_hash == hash_password(password):
            return True
    return False

# Signup function
def signup_user(username, password):
    users_data = load_user_data()
    if username not in users_data["username"].values:
        users_data = users_data.append({"username": username, "password": hash_password(password)}, ignore_index=True)
        save_user_data(users_data)
        return True
    return False

# UI and app flow logic
def show_homepage():
    st.title("SignX: Sign Language App")
    st.sidebar.header("Navigation")
    app_mode = st.sidebar.selectbox("Choose a page", ["Home", "Train", "Recognize", "Login", "Signup"])

    if app_mode == "Home":
        st.subheader("Welcome to SignX!")
        st.write("Learn sign language, train, and test your skills.")
    
    elif app_mode == "Train":
        st.subheader("ASL Training")
        sign_choice = st.selectbox("Choose a sign", list(ASL_ALPHABET.keys()))
        video_url = ASL_ALPHABET.get(sign_choice)
        st.video(video_url)
        
    elif app_mode == "Recognize":
        st.subheader("Live Sign Language Recognition")
        img, predicted = predict_asl()
        st.image(img)
        st.write(f"Predicted: {predicted}")

    elif app_mode == "Login":
        st.subheader("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if authenticate_user(username, password):
                st.success("Logged in successfully!")
            else:
                st.error("Invalid username or password")

    elif app_mode == "Signup":
        st.subheader("Signup")
        username = st.text_input("Choose a Username")
        password = st.text_input("Choose a Password", type="password")

        if st.button("Signup"):
            if signup_user(username, password):
                st.success("Signup successful!")
            else:
                st.error("Username already taken")


# Run the Streamlit app
if __name__ == "__main__":
    show_homepage()
