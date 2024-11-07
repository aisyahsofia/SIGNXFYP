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
    "Good Morning": f"{BASE_URL}GOODMORNING%20ASL.mp4",
    "Good Afternoon": f"{BASE_URL}GOODAFTERNOON%20ASL.mp4",
    "Good Evening": f"{BASE_URL}GOODEVENING%20ASL.mp4",
    "Good Night": f"{BASE_URL}GOODNIGHT%20ASL.mp4",
    "Thank You": f"{BASE_URL}THANKYOU%20ASL.mp4",
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
    
    camera_input = st.camera_input("Capture Image of your Sign")

    if camera_input is not None:
        image = cv2.imdecode(np.frombuffer(camera_input.getvalue(), np.uint8), 1)

        # Placeholder for model predictions
        # You can integrate a machine learning model here for sign recognition
        # For this example, let's assume the model recognized "Hello"
        detected_sign = "Hello"  # Placeholder for detected sign

        st.image(image, caption="Captured Sign", use_column_width=True)

        # Simulate progress tracking for the recognized sign
        if detected_sign:
            st.write(f"Detected sign: {detected_sign}")
            if st.button(f"Mark '{detected_sign}' as learned"):
                track_progress(st.session_state['username'], detected_sign)
                st.success(f"'{detected_sign}' marked as learned!")

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


help me with the keras. files and model training the machine to allow my app to detect the words on camera.

based on my machine learning codings:

# Collect Images

import os
import cv2

DATA_DIR = './data/static/images'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Set the number of classes and dataset size
number_of_classes = 25
dataset_size = 150

cap = cv2.VideoCapture(0)

# Loop for the specified number of classes
for j in range(number_of_classes):
    # Prompt the user to enter a new class name
    class_name = input(f'Enter the class name for dataset {j + 1} (will be stored in folder "): ').strip()
    class_dir = os.path.join(DATA_DIR, class_name)

    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Collecting data for class: {class_name}')

    # Wait for the user to be ready before starting collection
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        cv2.putText(frame, f'Press SPACE to start collecting data for class: {class_name}', 
                    (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            print("Exiting program.")
            cap.release()
            cv2.destroyAllWindows()
            exit()
        elif key & 0xFF == ord(' '):  # Spacebar to start collecting
            break

    # Collect dataset_size images for the current class
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        cv2.imshow('frame', frame)
        cv2.waitKey(25)

        # Save the image with a default naming convention
        image_filename = f"{counter}.jpg"
        cv2.imwrite(os.path.join(class_dir, image_filename), frame)
        counter += 1

        # Check for 'q' key press to quit during image collection
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            print("Data collection interrupted.")
            break

cap.release()
cv2.destroyAllWindows()

!pip install mediapipe scikit-learn tensorflow

!pip show opencv-python

# Collecting hand landmarks base on mediapipe x,y only

import os
import numpy as np
import mediapipe as mp
import cv2

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)

DATA_DIR = r'C:\Users\puter\OneDrive\Desktop\Sign-language-first\final\data\static\images'

static_landmarks = []
static_labels = []

# Count total images for progress tracking
total_images = sum(len(files) for _, _, files in os.walk(DATA_DIR))
processed_images = 0  # Initialize a counter for processed images

for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []
        x_ = []
        y_ = []

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
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

            static_landmarks.append(data_aux)
            static_labels.append(dir_)

        # Update and print progress
        processed_images += 1
        if processed_images % 10 == 0 or processed_images == total_images:  # Print every 10 images or last one
            print(f'Processed {processed_images}/{total_images} images.')

# Convert data and labels to NumPy arrays
static_landmarks = np.array(static_landmarks)
static_labels = np.array(static_labels)

# Save data and labels using NumPy
np.savez(r".\data\npz\compile.npz", static_landmarks=static_landmarks, static_labels=static_labels)
print('Done')

import os
import json

# Get unique labels
unique_labels = np.unique(static_labels)

# Create a dictionary with indices as keys and labels as values
label_dict = {str(index): label for index, label in enumerate(unique_labels)}

# Convert the dictionary to JSON format and save to a file
with open(r"C:\xampp\htdocs\Project\GitSIgn\Compile\data\labels\compile.json", 'w') as json_file:
    json.dump(label_dict, json_file, indent=4)

print("Labels saved to compile.json")

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# Load your data from .npz file
data_dict = np.load(r"C:\xampp\htdocs\Project\GitSIgn\Compile\data\npz\compile.npz")  # Ensure you have a .npz file
static_landmarks = data_dict['static_landmarks']  # Assuming 'data' is the key for features
static_labels = data_dict['static_labels']  # Assuming 'labels' is the key for labels

# Check the shape of the loaded data
print("Data shape:", static_landmarks.shape)
print("Labels shape before processing:", static_labels.shape)

# Ensure labels is a 1D array of class labels
# If your labels are originally strings like ['A', 'B', 'C', ...], convert them to integers
unique_labels, labels_indices = np.unique(static_labels, return_inverse=True)  # Convert to numerical indices
print("Unique labels:", unique_labels)
print("Labels indices shape:", labels_indices.shape)  # Should match the number of samples in data

# Convert labels to categorical (one-hot encoding)
num_classes = len(unique_labels)  # Number of unique classes
labels_categorical = to_categorical(labels_indices, num_classes=num_classes)  # Shape: (num_samples, num_classes)

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(static_landmarks, labels_categorical, test_size=0.2, shuffle=True)

# Define the Keras model
model = Sequential([
    Dense(128, activation='relu', input_shape=(static_landmarks.shape[1],)),  # Adjust input shape as needed
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')  # Output layer for number of classes
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=75, batch_size=32, validation_data=(x_test, y_test))



# Read json to use in live test

import json

# Load the JSON file
with open(r"C:\xampp\htdocs\Project\GitSIgn\Compile\data\labels\compile.json", 'r') as json_file:
    label_dict = json.load(json_file)

# Now label_dict contains your data
print(label_dict)  # This will print the loaded dictionary

# Read labels in the npz folders


import numpy as np

# Load your data from .npz file
data_dict = np.load(r"C:\xampp\htdocs\Project\GitSIgn\Compile\data\npz\compile.npz")  # Ensure you have a .npz file
# static_landmarks = data_dict['static_landmarks']  # Assuming 'data' is the key for features not needed currently
static_labels = data_dict['static_labels']  # Assuming 'labels' is the key for labels

# Assuming 'static_labels' is your array of labels from the npz file
# static_labels = np.array([...])  # Replace with the actual array

# Get unique sorted labels
unique_labels = sorted(np.unique(static_labels))

# Create dictionary with index as key and label as value
label_dict = {str(index): label for index, label in enumerate(unique_labels)}

print(label_dict)

# Live test

import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

# Load the Keras model this is needed if you start to run from keras (not needed if you just finished  training)
# model = load_model(r"C:\xampp\htdocs\Project\GitSIgn\Compile\data\keras\compile.keras")

# Initialize video capture
cap = cv2.VideoCapture(0)

# Set up MediaPipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        break  # Exit if the frame is not captured

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
        predicted_class_index = np.argmax(prediction, axis=1)[0] # e.g. 0 or 1 until max class total, in this case 25
        predicted_probability = prediction[0][predicted_class_index]  # Probability of the predicted class


        print(np.argmax(prediction, axis=1)[0])
        # Check if the probability is above the threshold (30%)
        if predicted_probability >= 0.3:
            # Convert to string to use as a key in the dictionary
            predicted_key = str(predicted_class_index)
            predicted_character = label_dict.get(predicted_key, 'Unknown')  # Use get to avoid KeyError
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

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit on 'q' key press
        break

cap.release()
cv2.destroyAllWindows()
