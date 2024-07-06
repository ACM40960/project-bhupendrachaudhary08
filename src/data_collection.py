# Import necessary packages/dependencies
import cv2
import logging
import mediapipe as mp
import numpy as np
import os
import time
from config.config import DATASET_PATH, NUM_SAMPLES, NUM_FRAMES, LOGGING_LEVEL, LOGGING_FORMAT
from utils import draw_styled_landmarks, extract_keypoints, load_labels, mediapipe_detection

# Configure logging settings
logging.basicConfig(level=LOGGING_LEVEL, format=LOGGING_FORMAT)

# Initialize MediaPipe modules for holistic model and drawing utilities
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Create a Mediapipe Holistic object for hand tracking and landmark extraction
holistic = mp_holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75)

# Create dataset directory if it does not exist
os.makedirs(DATASET_PATH, exist_ok=True)


def capture_gestures(label: str, num_samples: int = 100, num_frames: int = 30) -> bool:
    """
    Capture gesture data for a given label using webcam.
    :param label: The label for the gesture being captured.
    :param num_samples: Number of samples to capture.
    :param num_frames: Number of frames per sample.
    :return: True if the capture is interrupted, otherwise False.
    """
    # Initialise camera capture
    cap = cv2.VideoCapture(0)

    # Check if the camera is accessible
    if not cap.isOpened():
        # Log error message if camera is not accessible
        logging.error("Cannot access camera")
        return False
    # Flag to indicate if camera should be closed or not
    close_camera = False
    # Loop over the number of samples
    for sample in range(num_samples):
        logging.info(f"Starting collection for sample {sample + 1} of {num_samples}")

        # Flag to indicate if recording is in progress
        is_recording = False

        # Initialise the frame number
        frame_num = 0
        #  Loop until the required number of frames are captured or the camera is open
        while frame_num < num_frames and not close_camera:
            success, frame = cap.read()
            if not success:
                # Log a warning message if the frame was not captured successfully
                logging.warning("Ignoring empty camera frame.")
                continue

            # Perform MediaPipe detection on the captured frame
            image, results = mediapipe_detection(frame, holistic)

            # Draw the detected landmarks on the image
            draw_styled_landmarks(mp_drawing, mp_holistic, image, results)

            if not is_recording:
                cv2.putText(image, "Paused", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(image, f"Press 'Space' to start recording sample {sample + 1} for {label}", (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
            else:
                cv2.putText(image, f"Recording data for {label} - sample number {sample + 1}", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                # Extract keypoints from the detection results
                keypoints = extract_keypoints(results)
                try:
                    # Create the path for storing the sample
                    sample_path = os.path.join(DATASET_PATH, str(label), str(sample))

                    # Create the sample directory if it doesn't already exist
                    os.makedirs(sample_path, exist_ok=True)

                    # Create the path for storing the frame
                    frame_path = os.path.join(str(sample_path), f"{frame_num}.npy")

                    # Save the keypoints as a .npy file
                    np.save(frame_path, keypoints)
                    frame_num += 1
                except Exception as e:
                    # Log an error message if an exception occurs
                    logging.error(f"Error saving keypoints: {e}")
            cv2.imshow("MediaPipe Detection", image)

            key = cv2.waitKey(10) & 0xFF

            if key == ord(" "):
                # Wait for 1 second before starting recording
                time.sleep(1)
                is_recording = True

            if key == ord("q"):
                close_camera = True
                break

            # Break the loop if the camera window was closed
            if cv2.getWindowProperty("MediaPipe Detection", cv2.WND_PROP_VISIBLE) < 1:
                cap.release()
                cv2.destroyAllWindows()
                return True
        if close_camera:
            break
    cap.release()
    cv2.destroyAllWindows()
    return close_camera


def main() -> None:
    """
    Main function to start data collection for each label.
    """
    labels = load_labels()
    for key, label in labels.items():
        try:
            close_camera = capture_gestures(label, NUM_SAMPLES, NUM_FRAMES)
            if close_camera:
                break
        except Exception as e:
            logging.error(f"An error occurred while capturing gestures for {label}: {e}")



if __name__ == "__main__":
    main()
