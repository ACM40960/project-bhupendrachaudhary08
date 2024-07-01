# Load necessary packages/dependencies
import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize MediaPipe modules for holistic model and drawing utilities
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75)
mp_drawing = mp.solutions.drawing_utils


# Load labels from the file
def load_labels() -> dict[int, str]:
    with open("labels.txt", "r") as file:
        sign_labels: list[str] = [line.strip() for line in file.readlines()]
    return {idx: label for idx, label in enumerate(sign_labels)}


labels = load_labels()

# Create dataset directory
DATA_PATH = os.path.join('dataset')
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)


# Function to make detection
def mediapipe_detection(frame):
    try:
        # Color conversion BGR to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Making image non-writable
        image.flags.writeable = False
        # Make prediction
        results = holistic.process(image)
        # Making image writable
        image.flags.writeable = True
        # Color conversion RGB to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, results
    except Exception as e:
        print(f"Error in mediapipe_detection: {e}")
        return frame, None


def draw_styled_landmarks(image, results) -> None:
    try:
        # Draw face connections
        if results.face_landmarks:
            mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                                      mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                      mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))
        # Draw pose connections
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))
        # Draw left hand connections
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))
        # Draw right hand connections
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
    except Exception as e:
        print(f"Error in draw_styled_landmarks: {e}")


# Function to extract keypoints
def extract_keypoints(results):
    try:
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                         results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
        face = np.array([[res.x, res.y, res.z] for res in
                         results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
        left_hand = np.array([[res.x, res.y, res.z] for res in
                              results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(
            21 * 3)
        right_hand = np.array([[res.x, res.y, res.z] for res in
                               results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(
            21 * 3)
        return np.concatenate([pose, face, left_hand, right_hand])
    except Exception as e:
        print(f"Error in extract_keypoints: {e}")
        return np.zeros(33 * 4 + 468 * 3 + 21 * 3 + 21 * 3)


# Function to capture gestures
def capture_gestures(label, num_samples=100, num_frames=30):
    cap = cv2.VideoCapture(0)
    for sample in range(num_samples):
        for frame_num in range(num_frames):
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            image, results = mediapipe_detection(frame)

            # Draw landmarks on the image
            draw_styled_landmarks(image, results)

            # Wait for 1 second after each sample
            if frame_num == 0:
                cv2.putText(image, 'Starting collection', (120, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                cv2.putText(image, 'Collecting frames for {} video number {}'.format(label, sample), (15, 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.waitKey(1000)
            else:
                cv2.putText(image, 'Collecting frames for {} video number {}'.format(label, sample), (15, 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            keypoints = extract_keypoints(results)

            try:
                os.makedirs(os.path.join(DATA_PATH, label, str(sample)), exist_ok=True)
                frame_path = os.path.join(DATA_PATH, label, str(sample), str(frame_num))
                np.save(frame_path, keypoints)
            except Exception as e:
                print(f"Error saving keypoints: {e}")

            cv2.imshow('MediaPipe Detection', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


# Main Execution
def main() -> None:
    try:
        for label in labels.keys():
            capture_gestures(label, num_samples=100, num_frames=30)
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
