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


def capture_images_new(labels: dict[int, str], num_samples: int = 200) -> bool:
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
                existing_images = len(
                    [img for img in os.listdir(label_dir) if img.endswith(".jpg")])
                if existing_images > 0:
                    logging.error(f"Images already exist for label '{
                                  label}'. Either delete them or press ESC to skip.")
                    while True:
                        success, frame = cap.read()
                        if not success:
                            logging.warning("Ignoring empty camera frame.")
                            continue

                        cv2.putText(frame, f"Images exist for '{label}'. Delete them or press ESC to skip.",
                                    (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        cv2.imshow("Frame", frame)
                        interrupt = cv2.waitKey(10)

                        if interrupt & 0xFF == 27:  # ESC to skip this label
                            logging.info(
                                f"User chose to skip label '{label}'.")
                            break

                        if interrupt & 0xFF == ord('q'):  # Quit the program
                            logging.info("User pressed 'q'. Exiting program.")
                            cap.release()
                            cv2.destroyAllWindows()
                            return False

                    continue  # Skip this label if ESC was pressed
            remaining_images = num_samples
            current_hand = "Right Hand"
            half_samples = num_samples // 2
            allow_skip_or_quit = True
            logging.info(f"Starting capture for label '{label}'")
            interrupt = -1
            while remaining_images > 0:
                success, frame = cap.read()

                if not success:
                    # Log a warning message
                    logging.warning("Ignoring empty camera frame.")
                    continue

                # Display instructions on the frame
                if current_hand == "Right Hand":
                    instruction_text = f"Press SPACE to start capturing for '{
                        label}' for RIGHT hand or 'q' to quit."
                else:
                    instruction_text = "Switch to your LEFT hand and press SPACE to continue..."
                cv2.putText(frame, instruction_text, (20, frame.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                cv2.imshow("Frame", frame)

                interrupt = cv2.waitKey(10)

                if allow_skip_or_quit and (interrupt & 0xFF == ord('q') or interrupt & 0xFF == ord('Q')):
                    # Quit the program if 'q' is pressed
                    logging.info("User pressed 'q'. Exiting program.")
                    cap.release()
                    cv2.destroyAllWindows()
                    return False

                if allow_skip_or_quit and interrupt & 0xFF == 27:
                    # esc key
                    break

                if interrupt & 0xFF == ord(' '):
                    # Create directory for the label if it doesn't exist
                    if not os.path.exists(label_dir):
                        os.makedirs(label_dir, exist_ok=True)

                    logging.info(f"Starting capture for label '{label}'. Need {
                                 remaining_images} more images.")

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

                    # Capture images for the current hand
                    images_to_capture = half_samples if current_hand == "Right Hand" else remaining_images

                    # Capture the required number of images
                    while images_to_capture > 0:
                        success, frame = cap.read()
                        if not success:
                            logging.warning("Ignoring empty camera frame.")
                            continue

                        img_count = num_samples - remaining_images + 1

                        # Display the count of captured images on the main frame
                        cv2.putText(frame, f"Capturing image {img_count}/{num_samples} ({current_hand})",
                                    (20, frame.shape[0] -
                                     20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0),
                                    2, cv2.LINE_AA)
                        img_path = os.path.join(
                            label_dir, f"{num_samples - remaining_images + 1}.jpg")
                        cv2.imwrite(img_path, frame)
                        cv2.imshow("Frame", frame)
                        logging.info(f"Captured image {img_count} for label '{
                                     label}' ({current_hand})")
                        images_to_capture -= 1
                        remaining_images -= 1  # Decrease the remaining image count
                        # Small delay between captures to avoid duplicates
                        cv2.waitKey(100)
                    if current_hand == "Right Hand" and remaining_images > 0:
                        current_hand = "Left Hand"
                        allow_skip_or_quit = False  # Disallow skipping or quitting after right hand capture
            logging.info(f"Finished capturing {
                         num_samples} images for label '{label}'.")
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
