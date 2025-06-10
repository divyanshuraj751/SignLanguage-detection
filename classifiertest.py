import pickle
import cv2
import mediapipe as mp
import numpy as np

# ===== 1. Load Model =====
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']
labels_dict = {i: chr(65 + i) for i in range(26)}  # 0:A, 1:B, ..., 25:Z

# ===== 2. Initialize Camera =====
def initialize_camera():
    for camera_idx in [0, 1, 2]:  
        cap = cv2.VideoCapture(camera_idx)
        if cap.isOpened():
            print(f"Using camera index: {camera_idx}")
            return cap
    raise RuntimeError("No camera found!")

cap = initialize_camera()

# ===== 3. MediaPipe Setup =====
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ===== 4. Main Loop =====
while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Failed to capture frame")
        continue

    # Convert frame to RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks
            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                mp.solutions.drawing_styles.get_default_hand_connections_style()
            )

            # Extract and normalize landmarks
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)

            data_aux = []
            for lm in hand_landmarks.landmark:
                data_aux.append(lm.x - min_x)  # Normalized X
                data_aux.append(lm.y - min_y)  # Normalized Y

            # Predict gesture (ensure model outputs 0-25)
            prediction = model.predict([np.asarray(data_aux)])
            predicted_char = labels_dict[int(prediction[0])]

            # Draw bounding box and label
            h, w = frame.shape[:2]
            x1, y1 = int(min_x * w) - 10, int(min_y * h) - 10
            x2, y2 = int(max_x * w) + 10, int(max_y * h) + 10
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, predicted_char, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2, cv2.LINE_AA)

    # Display frame
    cv2.imshow('Sign Language Classifier (A-Z)', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

# ===== 5. Cleanup =====
cap.release()
cv2.destroyAllWindows()


