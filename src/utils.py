# Import necessary packages/dependencies
import cv2
import logging
import numpy as np
from typing import Tuple, Dict, Any

from config.config import LABELS_PATH


def load_labels() -> Dict[int, str]:
    """
    Load labels from the labels file.
    :return: Dictionary mapping indices to labels.
    """
    try:
        # Open the labels file and read all the lines
        with open(LABELS_PATH, "r") as file:
            sign_labels: list[str] = [line.strip() for line in file.readlines()]
            # Create a dictionary mapping indices to labels
        return {idx: label for idx, label in enumerate(sign_labels)}
    except Exception as e:
        # Log an error if an exception occurs
        logging.error(f"Error loading labels: {e}")
        return {}


def mediapipe_detection(frame: np.ndarray, holistic: Any) -> Tuple[np.ndarray, Any]:
    """
    Perform Mediapipe detection on a frame.

    :param frame: The input frame from the camera.
    :param holistic: The holistic model.
    :return: Processed image and detection results.
    """
    try:
        # Convert the frame from BGR to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Mark the image as non-writable to improve performance
        image.flags.writeable = False
        # Perform prediction with the holistic model
        results = holistic.process(image)
        # Mark the image as writable again
        image.flags.writeable = True
        # Convert the image back from RGB to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, results
    except Exception as e:
        # Log an error message if an exception occurs
        logging.error(f"Error in mediapipe_detection: {e}")
        return frame, None


def draw_styled_landmarks(mp_drawing: Any, mp_holistic: Any, image: np.ndarray, results: Any) -> None:
    """
    Draw landmarks on the frame

    :param mp_drawing: Mediapipe drawing utils.
    :param mp_holistic: Mediapipe holistic model.
    :param image: The image on which to draw.
    :param results: The detection results.
    """
    try:
        # Draw face landmarks
        if results.face_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.face_landmarks,
                mp_holistic.FACEMESH_CONTOURS,
                mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
            )

        # Draw pose landmarks
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
            )

        # Draw left hand landmarks
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
            )

        # Draw right hand landmarks
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )
    except Exception as e:
        # Log an error message if an exception occurs
        logging.error(f"Error in draw_styled_landmarks: {e}")


def extract_keypoints(results: Any) -> np.ndarray:
    """
    Extract keypoints from the Mediapipe detection results.

    :param results: Mediapipe detection results.
    :return: Flattened array of keypoints.
    """
    try:
        # Extract pose keypoints
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                         results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)

        # Extract face keypoints
        face = np.array([[res.x, res.y, res.z] for res in
                         results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)

        # Extract left hand keypoints
        left_hand = np.array([[res.x, res.y, res.z] for res in
                              results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(
            21 * 3)

        # Extract right hand keypoints
        right_hand = np.array([[res.x, res.y, res.z] for res in
                               results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(
            21 * 3)

        # Concatenate all keypoints into a single array and return
        return np.concatenate([pose, face, left_hand, right_hand])
    except Exception as e:
        # Log an error message if an exception occurs
        logging.error(f"Error in extract_keypoints: {e}")

        # Return an array of zeros if an error occurs
        return np.zeros(33 * 4 + 468 * 3 + 21 * 3 + 21 * 3)
