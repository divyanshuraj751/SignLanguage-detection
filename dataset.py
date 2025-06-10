import os
import pickle
import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'
data = []
labels = []

for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(dir_path):
        continue  # Skip non-folders

    for img_path in os.listdir(dir_path):
        img_file = os.path.join(dir_path, img_path)
        img = cv2.imread(img_file)
        if img is None:
            print(f"Warning: Could not read image {img_path}")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if not results.multi_hand_landmarks:
            continue  # Skip if no hand detected

        for hand_landmarks in results.multi_hand_landmarks:
            # Extract all landmarks first
            x_ = [lm.x for lm in hand_landmarks.landmark]
            y_ = [lm.y for lm in hand_landmarks.landmark]

            # Normalize landmarks relative to hand bounding box
            data_aux = []
            for x, y in zip(x_, y_):
                data_aux.append(x - min(x_))  # Normalize X
                data_aux.append(y - min(y_))  # Normalize Y

            data.append(data_aux)
            labels.append(dir_)

            # Optional: Visualize landmarks (debugging)
            annotated_img = img.copy()
            h, w, _ = img.shape
            for x, y in zip(x_, y_):
                cx, cy = int(x * w), int(y * h)  # Convert to pixel coordinates
                cv2.circle(annotated_img, (cx, cy), 5, (0, 255, 0), -1)
            cv2.imshow("Landmarks", annotated_img)
            cv2.waitKey(10)  # Short delay to show image

# Save data
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

cv2.destroyAllWindows()
print("Data saved to data.pickle")
