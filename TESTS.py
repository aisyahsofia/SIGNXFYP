import streamlit as st
import pandas as pd
import hashlib
import random
import cv2
import numpy as np
import os

# File paths
USERS_FILE = "users.csv"
PROGRESS_FILE = "progress.csv"
SIGN_DATA_FILE = "sign_language_data.csv"

# Base URL for GitHub raw files
BASE_URL = "https://raw.githubusercontent.com/aisyahsofia/SIGNXFYP/main/"

# ASL alphabet data (excluding J and Z)
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

# Training module
def training():
    st.subheader("Sign Language Training")
    for phrase, video in ASL_ALPHABET.items():
        st.write(f"Letter: {phrase}")
        try:
            st.video(video)
        except Exception as e:
            st.error(f"Error loading video: {str(e)}")
        if st.button(f"Mark {phrase} as learned"):
            track_progress(st.session_state['username'], phrase)

# Performance tracking
def track_progress(username, phrase):
    progress_data = load_progress_data()
    new_entry = pd.DataFrame([[username, phrase]], columns=["username", "phrase"])
    progress_data = pd.concat([progress_data, new_entry], ignore_index=True)
    save_progress_data(progress_data)
    st.success(f"'{phrase}' marked as learned!")

# Show progress
def show_progress(username):
    st.subheader("Your Learning Progress")
    progress_data = load_progress_data()
    user_progress = progress_data[progress_data['username'] == username]
    if user_progress.empty:
        st.write("No progress yet.")
    else:
        st.table(user_progress)

# Sign detection using webcam or file upload
def sign_detection():
    st.subheader("Sign Detection")

    # Option to upload an image or video
    uploaded_file = st.file_uploader("Upload Image or Video for Detection", type=["jpg", "png", "mp4"])

    if uploaded_file:
        if uploaded_file.type in ["jpg", "png"]:
            # Handle image for sign detection
            img = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(img, 1)
            detected_sign = detect_sign_in_image(img)
            st.image(img, caption="Uploaded Image", use_column_width=True)
            st.write(f"Detected Sign: {detected_sign}")
        elif uploaded_file.type == "mp4":
            # Handle video for sign detection
            video = uploaded_file.read()
            detected_sign = detect_sign_in_video(video)
            st.video(uploaded_file)
            st.write(f"Detected Sign: {detected_sign}")

# Mock function for detecting signs (only A to Y)
def detect_sign_in_image(image):
    # This function should be linked to an actual sign detection model
    # Simulating detection logic here by returning a random letter from A to Y
    detected_letter = random.choice(list(ASL_ALPHABET.keys()))
    return detected_letter

def detect_sign_in_video(video):
    # Simulating detection logic here for video (currently just returns a random letter)
    detected_letter = random.choice(list(ASL_ALPHABET.keys()))
    return detected_letter

# Main app layout
def main():
    st.sidebar.title("SignX App Menu")
    menu = ["Home", "Login", "Sign Up", "Training", "Progress", "Sign Detection"]
    choice = st.sidebar.radio("Select an option", menu)

    if choice == "Home":
        st.title("Welcome to SignX: Learn American Sign Language (ASL)")
        st.write("The app that helps you learn and communicate in ASL.")
    elif choice == "Login":
        login()
    elif choice == "Sign Up":
        sign_up()
    elif choice == "Training":
        if 'logged_in' in st.session_state:
            training()
        else:
            st.warning("You must be logged in to access this section.")
    elif choice == "Progress":
        if 'logged_in' in st.session_state:
            show_progress(st.session_state['username'])
        else:
            st.warning("You must be logged in to access this section.")
    elif choice == "Sign Detection":
        if 'logged_in' in st.session_state:
            sign_detection()
        else:
            st.warning("You must be logged in to access this section.")

if __name__ == "__main__":
    main()
