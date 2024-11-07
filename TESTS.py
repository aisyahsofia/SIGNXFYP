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

# Define the base directory where videos are stored
VIDEO_BASE_PATH = "C:/Users/puter/Videos/"  # Update this with your correct path

# Check if the path exists to avoid file errors
if not os.path.exists(VIDEO_BASE_PATH):
    st.error(f"The video base path does not exist: {VIDEO_BASE_PATH}")
    raise FileNotFoundError(f"Video base path not found: {VIDEO_BASE_PATH}")

# Sign language data for training
SIGN_LANGUAGE_DATA = {
    "Hello": os.path.join(VIDEO_BASE_PATH, "HELLO ASL.mp4"),
    "Good Morning": os.path.join(VIDEO_BASE_PATH, "GOODMORNING ASL.mp4"),
    "Good Afternoon": os.path.join(VIDEO_BASE_PATH, "GOODAFTERNOON ASL.mp4"),
    "Good Evening": os.path.join(VIDEO_BASE_PATH, "GOODEVENING ASL.mp4"),
    "Good Night": os.path.join(VIDEO_BASE_PATH, "GOODNIGHT ASL.mp4"),
    "Thank You": os.path.join(VIDEO_BASE_PATH, "THANKYOU.mp4"),
    "Sorry": os.path.join(VIDEO_BASE_PATH, "SORRY ASL.mp4"),
    "Please": os.path.join(VIDEO_BASE_PATH, "PLEASE ASL.mp4"),
    "Yes": os.path.join(VIDEO_BASE_PATH, "YES ASL.mp4"),
    "No": os.path.join(VIDEO_BASE_PATH, "NO ASL.mp4"),
    "How Are You?": os.path.join(VIDEO_BASE_PATH, "HOWAREYOU ASL.mp4"),
    "My Name Is...": os.path.join(VIDEO_BASE_PATH, "MYNAMEIS ASL.mp4"),
    "What Is Your Name?": os.path.join(VIDEO_BASE_PATH, "WHATISYOURNAME ASL.mp4"),
    "I Am Deaf": os.path.join(VIDEO_BASE_PATH, "IMDEAF ASL.mp4"),
    "I Am Hearing": os.path.join(VIDEO_BASE_PATH, "IMHEARING ASL.mp4"),
    "Where Is the Toilet?": os.path.join(VIDEO_BASE_PATH, "WHEREISTHETOILET ASL.mp4"),
    "Help me": os.path.join(VIDEO_BASE_PATH, "HELPME ASL.mp4"),
    "I Love You": os.path.join(VIDEO_BASE_PATH, "ILOVEYOU ASL.mp4"),
    "See You Later": os.path.join(VIDEO_BASE_PATH, "SEEYOULATER ASL.mp4"),
    "Goodbye": os.path.join(VIDEO_BASE_PATH, "GOODBYE ASL.mp4"),
}

# Basic ASL alphabet
ASL_ALPHABET = {
    'A': os.path.join(VIDEO_BASE_PATH, 'A ASL.mp4'),
    'B': os.path.join(VIDEO_BASE_PATH, 'B ASL.mp4'),
    'C': os.path.join(VIDEO_BASE_PATH, 'C ASL.mp4'),
    'D': os.path.join(VIDEO_BASE_PATH, 'D ASL.mp4'),
    'E': os.path.join(VIDEO_BASE_PATH, 'E ASL.mp4'),
    'F': os.path.join(VIDEO_BASE_PATH, 'F ASL.mp4'),
    'G': os.path.join(VIDEO_BASE_PATH, 'G ASL.mp4'),
    'H': os.path.join(VIDEO_BASE_PATH, 'H ASL.mp4'),
    'I': os.path.join(VIDEO_BASE_PATH, 'I ASL.mp4'),
    'J': os.path.join(VIDEO_BASE_PATH, 'J ASL.mp4'),
    'K': os.path.join(VIDEO_BASE_PATH, 'K ASL.mp4'),
    'L': os.path.join(VIDEO_BASE_PATH, 'L ASL.mp4'),
    'M': os.path.join(VIDEO_BASE_PATH, 'M ASL.mp4'),
    'N': os.path.join(VIDEO_BASE_PATH, 'N ASL.mp4'),
    'O': os.path.join(VIDEO_BASE_PATH, 'O ASL.mp4'),
    'P': os.path.join(VIDEO_BASE_PATH, 'P ASL.mp4'),
    'Q': os.path.join(VIDEO_BASE_PATH, 'Q ASL.mp4'),
    'R': os.path.join(VIDEO_BASE_PATH, 'R ASL.mp4'),
    'S': os.path.join(VIDEO_BASE_PATH, 'S ASL.mp4'),
    'T': os.path.join(VIDEO_BASE_PATH, 'T ASL.mp4'),
    'U': os.path.join(VIDEO_BASE_PATH, 'U ASL.mp4'),
    'V': os.path.join(VIDEO_BASE_PATH, 'V ASL.mp4'),
    'W': os.path.join(VIDEO_BASE_PATH, 'W ASL.mp4'),
    'X': os.path.join(VIDEO_BASE_PATH, 'X ASL.mp4'),
    'Y': os.path.join(VIDEO_BASE_PATH, 'Y ASL.mp4'),
    'Z': os.path.join(VIDEO_BASE_PATH, 'Z ASL.mp4')
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

# Quiz feature
def quiz():
    st.subheader("Sign Language Quiz")
    
    if 'current_question' not in st.session_state:
        st.session_state['current_question'] = random.choice(list(SIGN_LANGUAGE_DATA.keys()))

    question = st.session_state['current_question']
    
    st.write(f"What does this sign mean?")
    st.video(SIGN_LANGUAGE_DATA[question])

    answer = st.text_input("Your answer")

    if st.button("Submit"):
        if answer.strip().lower() == question.lower():
            st.success("Correct!")
            track_progress(st.session_state['username'], question)
            st.session_state['current_question'] = random.choice(list(SIGN_LANGUAGE_DATA.keys()))
        else:
            st.error(f"Incorrect! The correct answer was '{question}'.")

# Feedback system
def feedback():
    st.subheader("Feedback")
    feedback_text = st.text_area("Please provide your feedback or suggestions:")
    if st.button("Submit Feedback"):
        if feedback_text:
            st.success("Thank you for your feedback!")

# Main app flow
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if not st.session_state['logged_in']:
    st.sidebar.title("SignX: Next-Gen Technology for Deaf Communications")
    login_option = st.sidebar.selectbox("Login or Sign Up", ["Login", "Sign Up"])

    if login_option == "Login":
        login()
    else:
        sign_up()
else:
    st.sidebar.title(f"Welcome, {st.session_state['username']}")
    action = st.sidebar.selectbox("Action", ["Training", "ASL Alphabet", "Your Progress", "Quiz", "Sign Detection", "Feedback", "Logout"])

    if action == "Training":
        training()
    elif action == "ASL Alphabet":
        asl_alphabet_training()
    elif action == "Your Progress":
        show_progress(st.session_state['username'])
    elif action == "Quiz":
        quiz()
    elif action == "Sign Detection":
        sign_detection()
    elif action == "Feedback":
        feedback()
    elif action == "Logout":
        st.session_state['logged_in'] = False
        del st.session_state['username']
        st.write("You have been logged out.")
