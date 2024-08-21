import cv2
import logging
from matplotlib import pyplot as plt
import mediapipe as mp
import os
import pickle
import random

from config import ARTIFACTS_DIR, DATASET_DIR, DATA_PICKLE_FILENAME, LOGGING_FORMAT, LOGGING_LEVEL

# Configure logging settings
logging.basicConfig(level=LOGGING_LEVEL, format=LOGGING_FORMAT)

# Initialize MediaPipe Hands solution
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)


def augment_image(image):
    """Applies random transformations to an image for augmentation."""
    # Random horizontal flip
    if random.random() > 0.5:
        image = cv2.flip(image, 1)

    # Random rotation
    angle = random.randint(-15, 15)
    M = cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), angle, 1)
    image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    return image


def create_dataset() -> None:
    """
    Create a dataset from the captured images, extracting hand landmarks using MediaPipe.

    :return: None
    """
    try:
        data = []
        labels = []
        for label in os.listdir(DATASET_DIR):
            label_dir = os.path.join(DATASET_DIR, label)
            # Skip .DS_Store file or if it is not a directory
            if label == ".DS_Store" or not os.path.isdir(label_dir):
                continue
            for img_file in os.listdir(label_dir):
                if img_file.endswith(".jpg"):
                    logging.info(f"Processing image file: {img_file} for {label}")
                    # data_aux = []
                    # x_coords, y_coords, z_coords = [], [], []
                    image_path = os.path.join(label_dir, img_file)
                    image = cv2.imread(image_path)

                    # Augment original image
                    augmented_image = augment_image(image)
                    images_to_process = [image, augmented_image]  # Original and augmented
                    for img in images_to_process:
                        data_aux = []
                        x_coords, y_coords, z_coords = [], [], []
                        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
                            labels.append(label)
                        else:
                            logging.warning(f"No hand landmarks detected for {img_file} for {label}")
                    # plt.figure()
                    # plt.imshow(image_rgb)
                    # break
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)  # Ensure artifacts folder exists
        with open(os.path.join(ARTIFACTS_DIR, DATA_PICKLE_FILENAME), 'wb') as f:
            pickle.dump({'data': data, 'labels': labels}, f)
        logging.info(f"Dataset created and saved to {os.path.join(ARTIFACTS_DIR, DATA_PICKLE_FILENAME)}")
        # plt.show()
    except Exception as e:
        logging.error(f"An error occurred while creating the dataset: {e}")


def main() -> None:
    """
    Main function to start data creation process.
    """
    try:
        create_dataset()
    except Exception as e:
        print(f"An error occurred while capturing gestures: {e}")


if __name__ == "__main__":
    main()
