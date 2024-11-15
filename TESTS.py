import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

# Load the model (only once outside the loop)
model = load_model('data/keraspt1')

# Initialize video capture (change 0 if needed)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot access the camera.")
    exit()

# Set up MediaPipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)

# Load label dictionary from JSON (ensure you have the correct path)
with open(r"C:\xampp\htdocs\Project\GitSIgn\Compile\data\labels\compile.json", 'r') as json_file:
    label_dict = json.load(json_file)

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
        predicted_class_index = np.argmax(prediction, axis=1)[0]  # e.g. 0 or 1 until max class total, in this case 25
        predicted_probability = prediction[0][predicted_class_index]  # Probability of the predicted class

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

    # Show the frame with the prediction
    cv2.imshow('frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit on 'q' key press
        break

cap.release()
cv2.destroyAllWindows()
