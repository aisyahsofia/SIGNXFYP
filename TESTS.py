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

import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from math import dist

# Load Mediapipe hand tracking module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Define a function to detect ASL based on landmarks
def detect_asl_sign(landmarks):
    # Define ASL alphabet signs based on hand landmarks
    asl_alphabet = {
        0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H", 8: "I", 9: "K",
        10: "L", 11: "M", 12: "N", 13: "O", 14: "P", 15: "Q", 16: "R", 17: "S", 18: "T", 19: "U",
        20: "V", 21: "W", 22: "X", 23: "Y"
    }

    # Here you would compare the distance between key landmarks and match to the corresponding letter
    # In this example, we assume "A" is detected just as a placeholder.

    # Placeholder logic for detecting "A" (you need to adjust this to detect real signs)
    # Check if index finger and thumb are touching (like in ASL "A")
    if landmarks:
        thumb_tip = landmarks[4]  # Thumb tip (Landmark 4)
        index_tip = landmarks[8]  # Index tip (Landmark 8)

        # Compute Euclidean distance between thumb and index tip
        distance = dist((thumb_tip.x, thumb_tip.y), (index_tip.x, index_tip.y))
        
        # If the distance is small, we assume it's "A"
        if distance < 0.05:  # Adjust the threshold based on the distance between the fingers
            return "A"
        # Add more conditions for other letters here
    
    return "A"  # If the sign cannot be identified


# Camera feature for sign detection
def sign_detection():
    st.subheader("Sign Detection Camera")
    st.write("Point your camera to detect ASL signs.")

    camera_input = st.camera_input("Capture Image of your Sign")

    if camera_input is not None:
        image = cv2.imdecode(np.frombuffer(camera_input.getvalue(), np.uint8), 1)

        # Convert the image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image and get hand landmarks
        results = hands.process(image_rgb)

        # If hand landmarks are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get the list of landmarks as a numpy array
                landmarks = hand_landmarks.landmark

                # Get the detected ASL sign based on the landmarks
                detected_sign = detect_asl_sign(landmarks)

                # Draw the landmarks on the hand
                mp.solutions.drawing_utils.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Display the detected sign and image
                st.image(image, caption="Captured Sign", use_column_width=True)

                # Show the detected sign and provide option to track it
                if detected_sign:
                    st.write(f"Detected sign: {detected_sign}")
                    if st.button(f"Mark '{detected_sign}' as learned"):
                        track_progress(st.session_state['username'], detected_sign)
                        st.success(f"'{detected_sign}' marked as learned!")
        else:
            st.error("No hand detected. Please try again.")

    else:
        st.error("No image captured yet.")

# Quiz feature
def quiz():
    st.subheader("Sign Language Quiz")

    # Initialize quiz type and question if not set
    if 'quiz_type' not in st.session_state:
        st.session_state['quiz_type'] = random.choice(['word', 'alphabet'])

    if 'current_question' not in st.session_state:
        if st.session_state['quiz_type'] == 'word':
            st.session_state['current_question'] = random.choice(list(SIGN_LANGUAGE_DATA.keys()))
            st.session_state['question_data'] = SIGN_LANGUAGE_DATA
        else:
            st.session_state['current_question'] = random.choice(list(ASL_ALPHABET.keys()))
            st.session_state['question_data'] = ASL_ALPHABET

    # Display current question and video
    question = st.session_state['current_question']
    question_data = st.session_state['question_data']
    st.write("What does this sign mean?")
    st.video(question_data[question])

    # Display answer input and Submit button
    answer = st.text_input("Your answer")

    if 'submitted' not in st.session_state:
        st.session_state['submitted'] = False

    # Show feedback after Submit
    if st.button("Submit") and not st.session_state['submitted']:
        if answer.strip().lower() == question.lower():
            st.success("Correct!")
            track_progress(st.session_state['username'], question)
        else:
            st.error(f"Incorrect! The correct answer was '{question}'.")

        st.session_state['submitted'] = True  # Set submitted to True after submission

    # Show Next button after feedback is given
    if st.session_state['submitted'] and st.button("Next"):
        # Reset submitted state
        st.session_state['submitted'] = False

        # Select a new question and type
        st.session_state['quiz_type'] = random.choice(['word', 'alphabet'])
        if st.session_state['quiz_type'] == 'word':
            st.session_state['current_question'] = random.choice(list(SIGN_LANGUAGE_DATA.keys()))
            st.session_state['question_data'] = SIGN_LANGUAGE_DATA
        else:
            st.session_state['current_question'] = random.choice(list(ASL_ALPHABET.keys()))
            st.session_state['question_data'] = ASL_ALPHABET


# Feedback system
def feedback():
    st.subheader("Feedback")
    
    # Slider for rating (1-5 scale)
    rating = st.slider("Please rate your experience:", 1, 5, 3)  # Default to 3 (neutral)
    
    # Feedback text input
    feedback_text = st.text_area("Please provide your feedback or suggestions:")
    
    if st.button("Submit Feedback"):
        if feedback_text:
            st.success(f"Thank you for your feedback! You rated us {rating} out of 5.")
            # You can add logic here to save the feedback with the rating, e.g., saving to a CSV or a database.
        else:
            st.error("Please provide your feedback text.")
    

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