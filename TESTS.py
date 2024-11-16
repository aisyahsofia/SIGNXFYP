import streamlit as st
import pandas as pd
import hashlib
import os
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import gdown

print("gdown module loaded successfully.")


# Print OpenCV version for debugging
print(cv2.__version__)

# File paths for storing user and progress data
USERS_FILE = "users.csv"
PROGRESS_FILE = "progress.csv"
SIGN_DATA_FILE = "sign_language_data.csv"

# Base URL for GitHub raw files for sign language data
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
ASL_ALPHABET = {letter: f"{BASE_URL}{letter}%20ASL.mp4" for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'}

# Hashing function for passwords
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Save user data to a CSV
def save_user_data(users_data):
    users_data.to_csv(USERS_FILE, index=False)

# Load user data from a CSV
def load_user_data():
    try:
        return pd.read_csv(USERS_FILE)
    except FileNotFoundError:
        return pd.DataFrame(columns=["username", "password"])

# Save progress data to CSV
def save_progress_data(progress_data):
    progress_data.to_csv(PROGRESS_FILE, index=False)

# Load progress data from CSV
def load_progress_data():
    try:
        return pd.read_csv(PROGRESS_FILE)
    except FileNotFoundError:
        return pd.DataFrame(columns=["username", "phrase"])

# Login system
def login():
    st.title("SignX: Next-Gen Technology for Deaf Communications")
    users_data = load_user_data()
    st.subheader("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    hashed_password = hash_password(password)

    if st.button("Login"):
        if username in users_data['username'].values:
            stored_password = users_data[users_data['username'] == username]['password'].values[0]
            if stored_password == hashed_password:
                st.success(f"Welcome back, {username}!")
                st.session_state['logged_in'] = True
                st.session_state['username'] = username
            else:
                st.error("Invalid password")
        else:
            st.error("Username not found")

# Model download function
def download_model():
    model_path = "AisyahSignX59.h5"
    if not os.path.exists(model_path):  # Check if the model file exists locally
        gdown.download('https://drive.google.com/uc?id=1yRD3a942y5yID2atOF2o71lLwhOBoqJ-', model_path, quiet=False)
    return model_path

# Camera feature for sign detection
def sign_detection():
    st.subheader("Sign Detection Camera")
    st.write("Point your camera to detect ASL signs.")

    # Download and load the model
    model_path = download_model()
    model = load_model(model_path)

    # Setup MediaPipe hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    mp_draw = mp.solutions.drawing_utils

    # Create a placeholder for displaying video frames
    frame_placeholder = st.empty()

    # Open camera feed
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)

        # Convert the frame to RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

                # Flatten the landmarks to prepare input for the model
                data = np.array(landmarks.landmark).flatten().reshape(1, -1)
                prediction = model.predict(data)
                predicted_class = np.argmax(prediction, axis=1)  # Assuming the model returns class probabilities

                # Display the prediction result
                st.write(f"Prediction: Class {predicted_class}")

        # Update the displayed frame
        frame_placeholder.image(frame, channels="BGR", use_column_width=True)

    cap.release()

# Main function to handle the app flow
def main():
    st.sidebar.title("SignX Menu")
    menu_options = ["Login", "Sign Up", "Training", "ASL Alphabet", "Progress", "Sign Detection"]
    choice = st.sidebar.selectbox("Select an option", menu_options)

    if choice == "Login":
        login()
    elif choice == "Sign Up":
        sign_up()
    elif choice == "Training":
        if 'logged_in' in st.session_state and st.session_state['logged_in']:
            training()
        else:
            st.warning("Please log in first.")
    elif choice == "ASL Alphabet":
        if 'logged_in' in st.session_state and st.session_state['logged_in']:
            asl_alphabet_training()
        else:
            st.warning("Please log in first.")
    elif choice == "Progress":
        if 'logged_in' in st.session_state and st.session_state['logged_in']:
            show_progress(st.session_state['username'])
        else:
            st.warning("Please log in first.")
    elif choice == "Sign Detection":
        if 'logged_in' in st.session_state and st.session_state['logged_in']:
            sign_detection()
        else:
            st.warning("Please log in first.")

if __name__ == "__main__":
    main()
