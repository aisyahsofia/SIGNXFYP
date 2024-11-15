import streamlit as st
import pandas as pd
import time
import hashlib

# --- Helper Functions ---

# Simulated function to load user data from a CSV (for simplicity, using a temporary in-memory store)
def load_user_data():
    try:
        return pd.read_csv('users.csv')
    except FileNotFoundError:
        return pd.DataFrame(columns=['username', 'password'])

def save_user_data(users_data):
    users_data.to_csv('users.csv', index=False)

# Password hashing function
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Login system
def login():
    st.title("SignX: Next-Gen Technology for Deaf Communications")

    users_data = load_user_data()

    st.subheader("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in users_data['username'].values:
            stored_password = users_data[users_data['username'] == username]['password'].values[0]
            hashed_password = hash_password(password)
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

# --- Main App ---
def main():
    st.set_page_config(page_title="SignXTech - Learn ASL", page_icon=":guardsman:", layout="wide")
    
    # Sidebar Navigation
    menu = ["Login", "Training", "ASL Alphabets", "Quiz", "Your Progress", "Sign Detection", "Feedback"]
    choice = st.sidebar.selectbox("Navigate", menu)

    # Login page (Basic Authentication Simulation)
    if choice == "Login":
        login()

    # If logged in, show content; otherwise, show login page
    if st.session_state.get("logged_in", False):
        if choice == "Training":
            st.subheader("Training Modules")
            st.write("Module 1: Introduction to ASL")
            st.write("Module 2: Basic Signs")
            st.write("Module 3: Intermediate Signs")
            st.write("Module 4: Advanced Signs")
            st.write("Modules can be expanded with videos or interactive lessons.")

        elif choice == "ASL Alphabets":
            st.write("ASL Alphabet Guide")
            asl_alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
            for letter in asl_alphabet:
                st.write(f"Letter: {letter}")
                # Placeholder images for each ASL letter
                st.image(f"images/{letter}.jpg", caption=f"ASL for {letter}")

        elif choice == "Quiz":
            st.write("Test Your Knowledge with the ASL Quiz!")
            question = "What is the ASL sign for 'Hello'?"
            options = ["Wave", "Peace Sign", "Thumbs Up"]
            answer = st.radio(question, options)
            st.write(f"Your answer: {answer}")
            if answer == "Wave":
                st.success("Correct!")
            else:
                st.error("Incorrect, try again!")

        elif choice == "Your Progress":
            st.write("Your Learning Progress")
            progress_data = {
                'Module': ['Introduction to ASL', 'Basic Signs', 'Intermediate Signs', 'Advanced Signs'],
                'Completion': ['100%', '75%', '50%', '25%']
            }
            df = pd.DataFrame(progress_data)
            st.dataframe(df)

        elif choice == "Sign Detection":
            st.write("Sign Detection (using webcam):")
            # Real-time sign detection (could integrate a model in the future)
            cap = cv2.VideoCapture(0)
            stframe = st.empty()

            while True:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture image.")
                    break

                # Display webcam frame
                stframe.image(frame, channels="BGR", use_column_width=True)

                # Capture frame for further processing or sign recognition
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()

        elif choice == "Feedback":
            st.write("Provide Feedback on Your Learning Experience")
            rating = st.slider("Rate your experience (1-10)", 1, 10)
            feedback = st.text_area("Your Feedback", "Enter your feedback here...")
            if st.button("Submit Feedback"):
                with open('feedback.txt', 'a') as f:
                    f.write(f"{time.ctime()} - Rating: {rating} - Feedback: {feedback}\n")
                st.success("Thank you for your feedback!")

        # Logout
        if st.button("Log Out"):
            st.session_state.logged_in = False
            st.success("You have logged out successfully.")
    else:
        st.write("Please log in to access the app's features.")

# Run the app
if __name__ == "__main__":
    main()
