import os
import cv2
import numpy as np
import tensorflow as tf
from utils import draw_styled_landmarks, extract_keypoints, load_labels, mediapipe_detection
import mediapipe as mp
import logging
from config.config import NUM_FRAMES, MODEL_PATH, MODEL_NAME, LOGGING_LEVEL, LOGGING_FORMAT
from transformers import VitsTokenizer, VitsModel, set_seed
import torch
import sounddevice as sd

# Setup logging configuration
logging.basicConfig(level=LOGGING_LEVEL, format=LOGGING_FORMAT)

# Initialize MediaPipe modules for holistic model and drawing utilities
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Create a Mediapipe Holistic object for hand tracking and landmark extraction
holistic = mp_holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75)

# Load the pretrained Tensorflow model
model = tf.keras.models.load_model(os.path.join(MODEL_PATH, MODEL_NAME))
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Load the text-to-speech model and tokenizer
tts_tokenizer = VitsTokenizer.from_pretrained("facebook/mms-tts-eng")
tts_model = VitsModel.from_pretrained("facebook/mms-tts-eng")


def preprocess_frame_sequence(sequence):
    """
    Preprocess the sequence of frames for model prediction.

    :param sequence: List of keypoints for frames.
    :return: Preprocessed sequence with added batch dimension.
    """
    logging.info(f"Original sequence shape: {np.array(sequence).shape}")
    sequence = np.array(sequence)
    # sequence = sequence[..., np.newaxis]  # Add channel dimension
    # logging.info(f"Sequence shape after adding new axis: {sequence.shape}")
    sequence = np.expand_dims(sequence, axis=0)  # Add batch dimension
    logging.info(f"Sequence shape after adding batch dimension: {sequence.shape}")
    return sequence


def get_prediction(sequence):
    """
    Get model prediction for a given sequence.

    :param sequence: Preprocessed sequence of frames.
    :return: Prediction array, predicted label, and accuracy.
    """
    prediction = model.predict(sequence)[0]
    logging.info(f"Prediction array: {prediction}")
    predicted_label = np.argmax(prediction)
    accuracy = np.max(prediction)
    return prediction, predicted_label, accuracy


def prob_viz(labels, result, input_frame):
    """
    Visualize prediction probabilities on the input frame.

    :param labels: List of labels.
    :param result: Prediction array.
    :param input_frame: Original input frame.
    :return: Frame with probability visualization.
    """
    colours = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]
    output_frame = input_frame.copy()
    for num, prob in enumerate(result):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colours[num], -1)
        cv2.putText(output_frame, labels[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)
    return output_frame


def generate_speech(text):
    """
    Generate speech from text using Vits model.

    :param text: The text to be converted to speech.
    """
    inputs = tts_tokenizer(text=text, return_tensors="pt")
    set_seed(555)  # Make deterministic
    with torch.no_grad():
        outputs = tts_model(**inputs)
        print(outputs, "audio output")
    waveform = outputs.waveform[0]
    print(waveform, "output waveform")
    sampling_rate = tts_model.config.sampling_rate
    sd.play(waveform, samplerate=sampling_rate)
    sd.wait()


def capture_gestures(labels):
    """
    Capture gestures using webcam and predict using the trained model.

    :param labels: List of labels for the predictions.
    """
    # Access the camera
    cap = cv2.VideoCapture(0)
    sequence = []
    sentence = []
    predictions = []
    threshold = 0.8
    # Check if the camera is opened successfully
    if not cap.isOpened():
        logging.error("Cannot access camera")
        return
    while cap.isOpened():
        success, frame = cap.read()

        image, results = mediapipe_detection(frame, holistic)

        draw_styled_landmarks(mp_drawing, mp_holistic, image, results)

        # Prediction logic
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-NUM_FRAMES:]

        # Check if NUM_FRAMES frames have been accumulated
        if len(sequence) == NUM_FRAMES:
            processed_sequence = preprocess_frame_sequence(sequence)
            prediction, predicted_label, accuracy = get_prediction(processed_sequence)
            logging.info(f"Predicted Label: {predicted_label}, Accuracy: {accuracy}")
            predictions.append(predicted_label)
            if np.unique(predictions[-10:])[0] == predicted_label:
                if len(predictions) > 10:
                    # Check if the maximum predicted value is greater than the threshold
                    print("Predicted label: ", predicted_label)
                    predictions_10 = predictions[-10:]
                    most_common_pred = max(set(predictions_10), key=predictions.count)
                    logging.info(f"Most common prediction: {most_common_pred}")
                    if prediction[most_common_pred] > threshold:
                        print(' '.join(sentence), "Sentence")
                        if len(sentence) > 0:
                            if labels[most_common_pred] != sentence[-1]:
                                generate_speech(labels[predicted_label])
                                sentence.append(labels[most_common_pred])
                        else:
                            generate_speech(labels[predicted_label])
                            sentence.append(labels[most_common_pred])
            if len(sentence) > 5:
                sentence = sentence[-5:]
            image = prob_viz(labels, prediction, image)
        cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)

        cv2.imshow('Camera', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            # cap.release()
            break
    cap.release()
    cv2.destroyAllWindows()


# Main Execution
def main() -> None:
    try:
        labels = load_labels()
        capture_gestures(labels)
    except Exception as e:
        logging.error(f"An error occurred while capturing gestures: {e}")


if __name__ == "__main__":
    main()
