import pickle

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import os
from config.config import ARTIFACTS_PATH, DATA_PICKLE_NAME, DATASET_PATH, MODEL_NAME

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)


def create_dataset():
    data = []
    labels = []
    for dir_ in os.listdir(DATASET_PATH):
        if dir_ == ".DS_Store":  # Skip .DS_Store file
            continue
        dir_path = os.path.join(DATASET_PATH, dir_)
        for img_path in os.listdir(dir_path):
            if img_path == ".DS_Store":  # Skip .DS_Store file
                continue
            data_aux = []
            x_coords, y_coords, z_coords = [], [], []
            image = cv2.imread(os.path.join(DATASET_PATH, dir_, img_path))
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Perform prediction with the hands model
            result = hands.process(image_rgb)
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    # mp_drawing.draw_landmarks(
                    #     image_rgb,
                    #     hand_landmarks,
                    #     mp_hands.HAND_CONNECTIONS,
                    # )
                    for lm in hand_landmarks.landmark:
                        x_coords.append(lm.x)
                        y_coords.append(lm.y)
                        z_coords.append(lm.z)

                    x_min, y_min, z_min = min(x_coords), min(y_coords), min(z_coords)
                    for lm in hand_landmarks.landmark:
                        data_aux.extend([lm.x - x_min, lm.y - y_min, lm.z - z_min])
                data.append(data_aux)
                labels.append(dir_)
            # plt.figure()
            # plt.imshow(image_rgb)
            # break
    os.makedirs(ARTIFACTS_PATH, exist_ok=True)  # Ensure artifacts folder exists
    with open(os.path.join(ARTIFACTS_PATH, DATA_PICKLE_NAME), 'wb') as f:
        pickle.dump({'data': data, 'labels': labels}, f)
    # plt.show()


def main() -> None:
    """
    Main function to start data collection for each label.
    """
    try:
        create_dataset()
    except Exception as e:
        print(f"An error occurred while capturing gestures: {e}")


if __name__ == "__main__":
    main()
