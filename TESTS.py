import streamlit as st
import pandas as pd
import hashlib
import os
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import gdown

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

# Sign-up system
def sign_up():
    st.subheader("Sign Up")
    username = st.text_input("New Username")
    password = st.text_input("New Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")

    if st.button("Sign Up"):
        if password == confirm_password:
            users_data = load_user_data()
            if username not in users_data['username'].values:
                hashed_password = hash_password(password)
                new_user = pd.DataFrame([[username, hashed_password]], columns=["username", "password"])
                users_data = pd.concat([users_data, new_user], ignore_index=True)
                save_user_data(users_data)
                st.success("Account created successfully! Please log in.")
            else:
                st.error("Username already exists!")
        else:
            st.error("Passwords do not match")

# Training module with dropdown
def training():
    st.subheader("Sign Language Training")
    selected_phrase = st.selectbox("Choose a phrase to learn", list(SIGN_LANGUAGE_DATA.keys()))

    if selected_phrase:
        st.write(f"Phrase: {selected_phrase}")
        video_url = SIGN_LANGUAGE_DATA[selected_phrase]
        try:
            st.video(video_url)
        except Exception as e:
            st.error(f"Error loading video: {str(e)}")

        if st.button(f"Mark {selected_phrase} as learned"):
            track_progress(st.session_state['username'], selected_phrase)

# ASL alphabet training with dropdown
def asl_alphabet_training():
    st.subheader("Learn the ASL Alphabet")
    selected_letter = st.selectbox("Choose a letter to learn", list(ASL_ALPHABET.keys()))

    if selected_letter:
        st.write(f"Letter: {selected_letter}")
        video_url = ASL_ALPHABET[selected_letter]
        try:
            st.video(video_url)
        except Exception as e:
            st.error(f"Error loading video: {str(e)}")

        if st.button(f"Mark {selected_letter} as learned"):
            track_progress(st.session_state['username'], selected_letter)

# Performance tracking
def track_progress(username, phrase):
    progress_data = load_progress_data()
    new_entry = pd.DataFrame([[username, phrase]], columns=["username", "phrase"])
    progress_data = pd.concat([progress_data, new_entry], ignore_index=True)
    save_progress_data(progress_data)
    st.success(f"'{phrase}' marked as learned!")

# Display user progress
def show_progress(username):
    st.subheader("Your Learning Progress")
    progress_data = load_progress_data()
    user_progress = progress_data[progress_data['username'] == username]
    if user_progress.empty:
        st.write("No progress yet.")
    else:
        st.table(user_progress)

# Camera feature for sign detection
def sign_detection():
    st.subheader("Sign Detection Camera")
    st.write("Point your camera to detect ASL signs.")

    # Model download from Google Drive
    gdown.download('https://drive.google.com/uc?id=1yRD3a942y5yID2atOF2o71lLwhOBoqJ-', 'AisyahSignX59.h5', quiet=False)

    # Load the model
    model = load_model('AisyahSignX59.h5')

    # Setup MediaPipe hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    mp_draw = mp.solutions.drawing_utils

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

                # Get the landmarks and make predictions (model inference part is simplified)
                data = np.array(landmarks.landmark).flatten().reshape(1, -1)
                prediction = model.predict(data)

                st.write(f"Prediction: {prediction}")

        # Display the frame
        st.image(frame, channels="BGR")

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
