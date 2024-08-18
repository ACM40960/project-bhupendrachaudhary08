import pickle

import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from config.config import ARTIFACTS_PATH, MODEL_NAME, BASE_PATH
import os
from playsound import playsound
from threading import Thread

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)


def play_sound_for_prediction(prediction):
    """Plays the sound corresponding to the predicted alphabet."""
    sound_dir = os.path.join(BASE_PATH, 'audios')
    sound_file = os.path.join(sound_dir, f"{prediction}.wav")

    if os.path.exists(sound_file):
        # Play sound in a separate thread to avoid blocking
        Thread(target=playsound, args=(sound_file,), daemon=True).start()
    else:
        print(f"Sound file for '{prediction}' not found.")


def inference():
    with open(os.path.join(ARTIFACTS_PATH, MODEL_NAME), 'rb') as f:
        model_dict = pickle.load(f)
    model = model_dict['model']

    cap = cv2.VideoCapture(0)

    prediction_history = deque(maxlen=5)  # Queue to store the last 5 predictions
    last_prediction = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        x_coords, y_coords, z_coords = [], [], []
        data_aux = []

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,  # image to draw
                    hand_landmarks,  # model output
                    mp_hands.HAND_CONNECTIONS,  # hand connections
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                for lm in hand_landmarks.landmark:
                    x_coords.append(lm.x)
                    y_coords.append(lm.y)
                    z_coords.append(lm.z)
                x_min, y_min, z_min = min(x_coords), min(y_coords), min(z_coords)
                for lm in hand_landmarks.landmark:
                    data_aux.extend([lm.x - x_min, lm.y - y_min, lm.z - z_min])
            # Ensure the feature count matches the model's expected input size
            if len(data_aux) == model.n_features_in_:
                prediction = model.predict([np.asarray(data_aux)])
                predicted_label = prediction[0]

                # Append the current prediction to the history
                prediction_history.append(predicted_label)

                # Check if the last 5 predictions are the same
                if len(prediction_history) == 5 and all(
                        p == predicted_label for p in prediction_history) and predicted_label != last_prediction:
                    # Play the corresponding sound if last 5 predictions are the same
                    play_sound_for_prediction(predicted_label)
                    last_prediction = predicted_label
                    prediction_history.clear()  # Clear history after playing the sound

                # Determine if the predicted label exists in the labels_dict
                # if predicted_label in labels_dict.values():
                #     predicted_text = predicted_label
                # else:
                #     predicted_text = "Unknown"

                x1 = int(min(x_coords) * frame.shape[1]) - 10
                y1 = int(min(y_coords) * frame.shape[0]) - 10
                x2 = int(max(x_coords) * frame.shape[1]) + 10
                y2 = int(max(y_coords) * frame.shape[0]) + 10

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
                cv2.putText(frame, predicted_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
            else:
                print(f"Feature count mismatch: expected {model.n_features_in_}, got {len(data_aux)}")
                # Show an error message on the frame if the feature count is incorrect
                error_message = "Error: Feature count mismatch or Multiple hands detected"
                cv2.putText(frame, error_message, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                prediction_history.clear()
            # prediction = model.predict([np.asarray(data_aux)])
            # predicted_label = prediction[0]
            #
            # x1 = int(min(x_coords) * frame.shape[1])
            # y1 = int(min(y_coords) * frame.shape[0])
            # x2 = int(max(x_coords) * frame.shape[1])
            # y2 = int(max(y_coords) * frame.shape[0])

            # Show a message to the user on how to quit the program
        quit_message = "Press 'q' to quit"
        cv2.putText(frame, quit_message, (50, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0),2)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    inference()
