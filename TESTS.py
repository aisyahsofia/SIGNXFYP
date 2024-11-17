import streamlit as st
import pandas as pd
import hashlib
import random
import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
import mediapipe as mp
import requests


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

# Function to load sign language model
def load_sign_language_model():
    model_path = "C:/Users/puter/Downloads/final/data/keras/keras/AisyahSignX59.keras"
    
    try:
        model = load_model(model_path)
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Load the model
model = load_sign_language_model()

# Use the model for predictions if successfully loaded
if model:
    input_data = # prepare your input data here
    predictions = model.predict(input_data)
    print(predictions)

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
        for idx, row in user_progress.iterrows():
            st.write(f"- {row['phrase']}")

# App structure
def main():
    st.sidebar.title("SignX Menu")
    menu = ["Home", "Login", "Sign Up", "Training", "ASL Alphabet Training", "Progress", "Logout"]
    choice = st.sidebar.radio("Select an option", menu)

    if choice == "Home":
        st.title("Welcome to SignX!")
        st.write("This app helps you learn American Sign Language (ASL).")
    
    elif choice == "Login":
        login()
    
    elif choice == "Sign Up":
        sign_up()

    elif choice == "Training":
        if "logged_in" in st.session_state:
            training()
        else:
            st.error("Please log in first")

    elif choice == "ASL Alphabet Training":
        if "logged_in" in st.session_state:
            asl_alphabet_training()
        else:
            st.error("Please log in first")

    elif choice == "Progress":
        if "logged_in" in st.session_state:
            show_progress(st.session_state['username'])
        else:
            st.error("Please log in first")

    elif choice == "Logout":
        st.session_state.clear()
        st.success("Logged out successfully")

if __name__ == "__main__":
    main()
