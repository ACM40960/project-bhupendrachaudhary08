import cv2
import logging
import os
import time

from config import DATASET_DIR, LOGGING_FORMAT, LOGGING_LEVEL, NUM_SAMPLES
from utils import load_labels

# Configure logging settings
logging.basicConfig(level=LOGGING_LEVEL, format=LOGGING_FORMAT)

# Create dataset directory if it does not exist
os.makedirs(DATASET_DIR, exist_ok=True)


def capture_images_new(labels: dict[int, str], num_samples: int = 50) -> bool:
    """
    Capture images for each label using the webcam and store them in the dataset directory.
    :param labels: Dictionary mapping indices to labels (int -> str).
    :param num_samples: Number of images to capture per label (default is 100).
    :return: True if images were successfully captured, False otherwise.
    """
    try:
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            logging.error("Cannot access camera")
            return False
        for key, label in labels.items():
            label_dir = os.path.join(DATASET_DIR, label)

            # Check if the directory exists to determine existing and remaining images
            if os.path.exists(label_dir):
                # Determine how many images already exist
                existing_images = len([img for img in os.listdir(label_dir) if img.endswith(".jpg")])
                remaining_images = num_samples - existing_images
            else:
                existing_images = 0
                remaining_images = num_samples

            if remaining_images <= 0:
                logging.info(f"Already have {num_samples} images for label '{label}'. Skipping...")
                continue

            logging.info(f"Starting capture for label '{label}'. Need {remaining_images} more images.")
            interrupt = -1
            while remaining_images > 0:
                success, frame = cap.read()

                if not success:
                    logging.warning("Ignoring empty camera frame.")  # Log a warning message
                    continue

                # Display instructions on the frame
                instruction_text = f"Press SPACE to start capturing for label '{label}' or 'q' to quit or 'esc' to skip."
                cv2.putText(frame, instruction_text, (20, frame.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                cv2.imshow("Frame", frame)

                interrupt = cv2.waitKey(10)

                if interrupt & 0xFF == ord('q'):
                    # Quit the program if 'q' is pressed
                    logging.info("User pressed 'q'. Exiting program.")
                    cap.release()
                    cv2.destroyAllWindows()
                    return False

                if interrupt & 0xFF == 27:
                    # esc key
                    break

                if interrupt & 0xFF == ord(' '):
                    # Create directory for the label if it doesn't exist
                    if not os.path.exists(label_dir):
                        os.makedirs(label_dir, exist_ok=True)

                    logging.info(f"Starting capture for label '{label}'. Need {remaining_images} more images.")

                    # Countdown before capturing starts
                    start_time = time.time()
                    while time.time() - start_time < 3:
                        success, frame = cap.read()
                        print(time.time() - start_time, "Inside the countdown")
                        if not success:
                            logging.warning("Ignoring empty camera frame.")
                            continue

                        countdown = 3 - int(time.time() - start_time)
                        cv2.putText(frame, f"Starting in {countdown}...", (20, frame.shape[0] - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
                        cv2.imshow("Frame", frame)

                        cv2.waitKey(100)  # Small delay for display

                    # Capture the required number of images
                    while remaining_images > 0:
                        success, frame = cap.read()
                        if not success:
                            logging.warning("Ignoring empty camera frame.")
                            continue

                        # Display the count of captured images on the main frame
                        cv2.putText(frame, f"Capturing image {num_samples - remaining_images + 1}/{num_samples}",
                                    (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0),
                                    2, cv2.LINE_AA)
                        img_path = os.path.join(label_dir, f"{num_samples - remaining_images + 1}.jpg")
                        cv2.imwrite(img_path, frame)
                        cv2.imshow("Frame", frame)
                        logging.info(f"Captured image {num_samples - remaining_images + 1} for label '{label}'")
                        remaining_images -= 1  # Decrease the remaining image count
                        cv2.waitKey(100)  # Small delay between captures to avoid duplicates
                    logging.info(f"Finished capturing {num_samples} images for label '{label}'.")
            if interrupt & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        return True
    except Exception as e:
        logging.error(f"An error occurred during image capture: {e}")
        return False


def main() -> None:
    """
    Main function to start data collection for each label.
    """
    labels = load_labels()
    if not labels:
        logging.error("No labels found. Exiting.")
        return
    try:
        capture_images_new(labels, NUM_SAMPLES)
    except Exception as e:
        logging.error(f"An error occurred while capturing gestures: {e}")


if __name__ == "__main__":
    main()
