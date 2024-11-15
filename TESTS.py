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

# Sign Detection Camera Feature
def sign_detection():
    st.subheader("Sign Detection Camera")
    st.write("Point your camera to detect ASL signs.")

    camera_input = st.camera_input("Capture Image of your Sign")

    if camera_input is not None:
        image = cv2.imdecode(np.frombuffer(camera_input.getvalue(), np.uint8), 1)

        # Display the captured image
        st.image(image, caption="Captured Sign", use_column_width=True)

        # Placeholder: Image comparison for predefined signs
        detected_sign = detect_sign(image)

        # Display the detected sign
        if detected_sign:
            st.write(f"Detected sign: {detected_sign}")
            if st.button(f"Mark '{detected_sign}' as learned"):
                track_progress(st.session_state['username'], detected_sign)
                st.success(f"'{detected_sign}' marked as learned!")
    else:
        st.error("No image captured yet.")

# Function to detect the sign from the captured image
def detect_sign(image):
    # Example: Match the captured image with predefined signs using basic feature matching
    # For simplicity, this part assumes you have a set of predefined images for each sign

    # Predefined images (in this case, for "Hello" sign)
    hello_sign_image = cv2.imread("path_to_hello_sign_image.jpg")  # Replace with actual image path
    similarity = compare_images(image, hello_sign_image)

    if similarity > 0.8:  # If similarity is high, mark it as "Hello"
        return "Hello"
    # Add more signs and matching conditions as needed
    return None

# Function to compare two images and return a similarity score
def compare_images(image1, image2):
    # Convert images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Use ORB feature matching
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    # Use brute force matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Calculate similarity score (lower matches = better match)
    similarity = len(matches) / max(len(kp1), len(kp2))  # Example similarity measure
    return similarity

# User Registration Page
def register_user():
    st.subheader("Register New User")
    username = st.text_input("Enter a username")
    password = st.text_input("Enter a password", type='password')
    confirm_password = st.text_input("Confirm your password", type='password')

    if st.button("Register"):
        if password == confirm_password:
            users_data = load_user_data()
            if username not in users_data["username"].values:
                hashed_password = hash_password(password)
                new_user = pd.DataFrame({"username": [username], "password": [hashed_password]})
                users_data = pd.concat([users_data, new_user], ignore_index=True)
                save_user_data(users_data)
                st.success(f"User '{username}' successfully registered!")
            else:
                st.error(f"Username '{username}' is already taken.")
        else:
            st.error("Passwords do not match.")

# User Login Page
def login_user():
    st.subheader("Login")
    username = st.text_input("Enter your username")
    password = st.text_input("Enter your password", type='password')

    if st.button("Login"):
        users_data = load_user_data()
        if username in users_data["username"].values:
            stored_password = users_data[users_data["username"] == username]["password"].values[0]
            if hash_password(password) == stored_password:
                st.session_state["username"] = username
                st.success(f"Welcome back, {username}!")
                main_app()
            else:
                st.error("Incorrect password.")
        else:
            st.error("Username not found.")

# Main App Page
def main_app():
    st.sidebar.subheader("Navigation")
    option = st.sidebar.selectbox("Choose an action", ["Home", "Learn Signs", "Track Progress", "Sign Detection", "Logout"])

    if option == "Home":
        show_home_page()
    elif option == "Learn Signs":
        learn_signs()
    elif option == "Track Progress":
        track_progress_page()
    elif option == "Sign Detection":
        sign_detection()
    elif option == "Logout":
        st.session_state["username"] = None
        st.success("You have logged out.")

# Show Home Page
def show_home_page():
    st.title("Welcome to SignXTech App!")
    st.write("This app will help you learn American Sign Language (ASL).")

# Learn Signs
def learn_signs():
    st.subheader("Learn ASL Signs")
    sign = st.selectbox("Select a sign to learn", list(SIGN_LANGUAGE_DATA.keys()))
    if sign:
        st.video(SIGN_LANGUAGE_DATA[sign])

# Track Progress Page
def track_progress_page():
    st.subheader("Track Your Progress")
    if "username" not in st.session_state:
        st.error("Please login first.")
        return

    username = st.session_state["username"]
    progress_data = load_progress_data()

    # Show progress for the logged-in user
    user_progress = progress_data[progress_data["username"] == username]
    if user_progress.empty:
        st.write("No progress yet.")
    else:
        st.write("Learned Signs:")
        st.write(user_progress)

# Function to track progress
def track_progress(username, learned_sign):
    progress_data = load_progress_data()

    # Check if the user already has progress data
    if username not in progress_data["username"].values:
        new_progress = pd.DataFrame({"username": [username], "phrase": [learned_sign]})
        progress_data = pd.concat([progress_data, new_progress], ignore_index=True)
    else:
        progress_data.loc[progress_data["username"] == username, "phrase"] = learned_sign

    save_progress_data(progress_data)

# Display progress
def show_progress():
    if "username" not in st.session_state:
        st.error("Please login to view your progress.")
        return

    username = st.session_state["username"]
    progress_data = load_progress_data()

    user_progress = progress_data[progress_data["username"] == username]
    if not user_progress.empty:
        st.write(f"Progress for {username}:")
        st.write(user_progress)
    else:
        st.write("No progress tracked yet.")

# Streamlit App Entry Point
def run_app():
    if "username" in st.session_state:
        main_app()
    else:
        login_user()

# Run the app
if __name__ == "__main__":
    run_app()

