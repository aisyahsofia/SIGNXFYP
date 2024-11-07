import streamlit as st
import pandas as pd
import hashlib
import random
import cv2
import numpy as np
import os

print(cv2.__version__)

# File paths
USERS_FILE = "users.csv"
PROGRESS_FILE = "progress.csv"
SIGN_DATA_FILE = "sign_language_data.csv"

# Sign language data for training
SIGN_LANGUAGE_DATA = {
    "Hello": r"C:\Users\puter\Downloads\HELLO ASL.mp4",
    "Good Morning": r"C:\Users\puter\Downloads\GOODMORNING ASL.mp4",
    "Good Afternoon": r"C:\Users\puter\Downloads\GOODAFTERNOON ASL.mp4",
    "Good Evening": r"C:\Users\puter\Downloads\GOODEVENING ASL.mp4",
    "Good Night": r"C:\Users\puter\Downloads\GOODNIGHT ASL.mp4",
    "Thank You": r"C:\Users\puter\Downloads\THANKYOU ASL.mp4",
    "Sorry": r"C:\Users\puter\Downloads\SORRY ASL.mp4",
    "Please": r"C:\Users\puter\Downloads\PLEASE ASL.mp4",
    "Yes": r"C:\Users\puter\Downloads\YES ASL.mp4",
    "No": r"C:\Users\puter\Downloads\NO ASL.mp4",
    "How Are You?": r"C:\Users\puter\Downloads\HOWAREYOU ASL.mp4",
    "My Name Is...": r"C:\Users\puter\Downloads\MYNAMEIS ASL.mp4",
    "What Is Your Name?": r"C:\Users\puter\Downloads\WHATISYOURNAME ASL.mp4",
    "I Am Deaf": r"C:\Users\puter\Downloads\IMDEAF ASL.mp4",
    "I Am Hearing": r"C:\Users\puter\Downloads\IMHEARING ASL.mp4",
    "Where Is the Toilet?": r"C:\Users\puter\Downloads\WHEREISTHETOILET ASL.mp4",
    "Help me": r"C:\Users\puter\Downloads\HELPME ASL.mp4",
    "I Love You": r"C:\Users\puter\Downloads\ILOVEYOU ASL.mp4",
    "See You Later": r"C:\Users\puter\Downloads\SEEYOULATER ASL.mp4",
    "Goodbye": r"C:\Users\puter\Downloads\GOODBYE ASL.mp4",
}

# Basic ASL alphabet
ASL_ALPHABET = {
    'A': r"C:\Users\puter\Downloads\A ASL.mp4",
    'B': r"C:\Users\puter\Downloads\B ASL.mp4",
    'C': r"C:\Users\puter\Downloads\C ASL.mp4",
    'D': r"C:\Users\puter\Downloads\D ASL.mp4",
    'E': r"C:\Users\puter\Downloads\E ASL.mp4",
    'F': r"C:\Users\puter\Downloads\F ASL.mp4",
    'G': r"C:\Users\puter\Downloads\G ASL.mp4",
    'H': r"C:\Users\puter\Downloads\H ASL.mp4",
    'I': r"C:\Users\puter\Downloads\I ASL.mp4",
    'J': r"C:\Users\puter\Downloads\J ASL.mp4",
    'K': r"C:\Users\puter\Downloads\K ASL.mp4",
    'L': r"C:\Users\puter\Downloads\L ASL.mp4",
    'M': r"C:\Users\puter\Downloads\M ASL.mp4",
    'N': r"C:\Users\puter\Downloads\N ASL.mp4",
    'O': r"C:\Users\puter\Downloads\O ASL.mp4",
    'P': r"C:\Users\puter\Downloads\P ASL.mp4",
    'Q': r"C:\Users\puter\Downloads\Q ASL.mp4",
    'R': r"C:\Users\puter\Downloads\R ASL.mp4",
    'S': r"C:\Users\puter\Downloads\S ASL.mp4",
    'T': r"C:\Users\puter\Downloads\T ASL.mp4",
    'U': r"C:\Users\puter\Downloads\U ASL.mp4",
    'V': r"C:\Users\puter\Downloads\V ASL.mp4",
    'W': r"C:\Users\puter\Downloads\W ASL.mp4",
    'X': r"C:\Users\puter\Downloads\X ASL.mp4",
    'Y': r"C:\Users\puter\Downloads\Y ASL.mp4",
    'Z': r"C:\Users\puter\Downloads\Z ASL.mp4"
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
    for phrase, video in SIGN_LANGUAGE_DATA.items():
        st.write(f"Phrase: {phrase}")
        try:
            st.video(video)
        except Exception as e:
            st.error(f"Error loading video: {str(e)}")
        if st.button(f"Mark {phrase} as learned"):
            track_progress(st.session_state['username'], phrase)

# ASL alphabet training
def asl_alphabet_training():
    st.subheader("Learn the ASL Alphabet")
    for letter, video in ASL_ALPHABET.items():
        st.write(f"Letter: {letter}")
        try:
            st.video(video)
        except Exception as e:
            st.error(f"Error loading video: {str(e)}")
        if st.button(f"Mark {letter} as learned"):
            track_progress(st.session_state['username'], letter)

# Performance tracking
def track_progress(username, phrase):
    progress_data = load_progress_data()
    new_entry = pd.DataFrame([[username, phrase]], columns=["username", "phrase"])
    progress_data = pd.concat([progress_data, new_entry], ignore_index=True)
    save_progress_data(progress_data)
    st.success(f"{phrase} marked as learned!")

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
    
    camera_input = st.camera_input("Capture Image of your Sign")

    if camera_input is not None:
        image = cv2.imdecode(np.frombuffer(camera_input.getvalue(), np.uint8), 1)
        
        # Placeholder for model predictions
        st.write("This feature requires a model for sign detection.")
    else:
        st.error("No image captured yet.")

# Main app logic
def app():
    if 'logged_in' not in st.session_state or not st.session_state['logged_in']:
        login()
        sign_up()
    else:
        username = st.session_state['username']
        st.sidebar.title("Menu")
        menu_options = ["Training", "ASL Alphabet", "Progress", "Sign Detection"]
        choice = st.sidebar.radio("Select an option", menu_options)

        if choice == "Training":
            training()
        elif choice == "ASL Alphabet":
            asl_alphabet_training()
        elif choice == "Progress":
            show_progress(username)
        elif choice == "Sign Detection":
            sign_detection()

# Run the app
if __name__ == "__main__":
    app()
