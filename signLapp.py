import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np
from gtts import gTTS
import os
import kaggle
import zipfile
import shutil
import string
import pyttsx3
import time
import threading

# Initialize global model variable
model = None
gesture_mapping = {i: letter for i, letter in enumerate(string.ascii_uppercase)}

last_gesture = None
last_speech_time = time.time()

def text_to_speech(text):
    """Speak the text by creating a new engine instance in each function call."""
    def speak():
        # Initialize the engine locally within the function
        local_engine = pyttsx3.init()
        local_engine.say(text)
        local_engine.runAndWait()
        local_engine.stop()

    # Run TTS in a separate thread to avoid blocking Streamlit
    tts_thread = threading.Thread(target=speak)
    tts_thread.start()
    
# Function to download dataset from Kaggle with progress
def download_kaggle_dataset(dataset_name, download_path='datasets'):
    """Download dataset from Kaggle with progress using Kaggle API."""
    if not os.path.exists(download_path):
        os.makedirs(download_path)
        
    # Initialize the progress bar
    with st.spinner("Downloading dataset..."):
        kaggle.api.dataset_download_files(dataset_name, path=download_path, unzip=True)
    
    st.success(f"Dataset {dataset_name} downloaded successfully!")

# Create a CNN model for sign language classification
def create_model(num_classes):
    """Define a CNN model for sign language classification."""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')  # Adjust to match the actual number of classes
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train_model(data_path):
    """Train the model on the provided dataset."""
    # Use the number of classes based on the data generator
    train_datagen = ImageDataGenerator(rescale=1.0 / 255)
    train_data = train_datagen.flow_from_directory(
        data_path,
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical'  # Ensures labels are one-hot encoded
    )
    num_classes = len(train_data.class_indices)
    
    global model
    model = create_model(num_classes=num_classes)

    # Progress bar for training
    progress_bar = st.progress(0)
    epochs = 5
    
    # Train the model
    for epoch in range(epochs):
        model.fit(train_data, epochs=1)
        progress_bar.progress((epoch + 1) / epochs)
    
    model.save('sign_language_model.h5')
    progress_bar.empty()  # Clear the progress bar
    st.success("Model trained and saved successfully!")


def load_model():
    """Load the model from .h5 file."""
    global model
    with st.spinner("Loading model..."):
        try:
            model = tf.keras.models.load_model('sign_language_model.h5')
            st.success("Model loaded successfully!")
        except Exception as e:
            st.warning("No pre-trained model found. Please train a model first.")
            print("Error loading model:", e)

def predict_gesture(frame):
    """Predict the gesture from a camera frame using the loaded model."""
    resized_frame = cv2.resize(frame, (64, 64))  # Resize to match model's input shape
    resized_frame = resized_frame / 255.0  # Normalize the image
    
    # Predict and print the model output for debugging
    prediction = model.predict(np.expand_dims(resized_frame, axis=0))
    predicted_class = np.argmax(prediction)
    confidence = prediction[0][predicted_class]
    
    return predicted_class, confidence


# Streamlit App
st.title("Sign Language Detection and Training App")

# Sidebar Menu
option = st.sidebar.selectbox("Choose an option:", ["Home", "Train Model", "Sign Language Detection", "Download Dataset"])

# Download dataset from Kaggle if selected
if option == "Download Dataset":
    dataset_name = "grassknoted/asl-alphabet"  # ASL Alphabet Dataset on Kaggle
    download_kaggle_dataset(dataset_name, download_path='datasets')

elif option == "Train Model":
    st.header("Train the Model")
    
    # After downloading the dataset, allow the user to specify the data path
    data_path = 'datasets/asl_alphabet_train/asl_alphabet_train'  # Adjust the path according to the dataset structure
    if st.button("Train Model"):
        train_model(data_path)

elif option == "Sign Language Detection":
    st.header("Sign Language Detection")

    # Load the pre-trained model
    load_model()

    # Camera feed with prediction
    run_camera = st.checkbox("Open Camera")
    FRAME_WINDOW = st.image([])

    # Placeholder for detected gesture display
    gesture_display = st.empty()

    if run_camera and model is not None:
        cap = cv2.VideoCapture(1)  # Use 0 for default camera if 1 doesn't work

        if cap.isOpened():
            while run_camera:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to access the camera.")
                    break

                # Show camera feed
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                FRAME_WINDOW.image(frame_rgb)

                # Predict gesture and get confidence
                gesture, confidence = predict_gesture(frame_rgb)
                gesture_text = gesture_mapping.get(gesture, "Unknown gesture")

                # Display detected gesture
                gesture_display.subheader(f"Detected Gesture: {gesture_text} (Confidence: {confidence:.2f})")

            cap.release()
        else:
            st.error("Failed to open the camera.")
else:
    st.write("Select an option from the sidebar to start.")
