# Import necessary packages/dependencies
import logging
import numpy as np
import os
from sklearn.metrics import classification_report, multilabel_confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense

from config.config import DATASET_PATH, NUM_FRAMES, MODEL_PATH, MODEL_NAME, LOGS_PATH, LOGGING_LEVEL, LOGGING_FORMAT
from utils import load_labels

# Configure logging
logging.basicConfig(level=LOGGING_LEVEL, format=LOGGING_FORMAT)


def load_dataset(sign_labels):
    """
    Load dataset from the saved keypoints.

    :param sign_labels: Dictionary mapping labels to indices.
    :return: Numpy arrays of landmarks and labels.
    """
    try:
        landmarks, labels = [], []
        for index, label in sign_labels.items():
            label_path = os.path.join(DATASET_PATH, str(label))
            if os.path.isdir(label_path):
                for sample in os.listdir(label_path):
                    sample_path = os.path.join(label_path, sample)
                    frames = [np.load(os.path.join(sample_path, frame)) for frame in sorted(os.listdir(sample_path))]
                    if len(frames) == NUM_FRAMES:
                        landmarks.append(frames)
                        labels.append(index)
        landmarks, labels = np.array(landmarks), np.array(labels)
        return landmarks, labels
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return None, None


# Prepare data
def prepare_data(landmarks, labels, num_classes):
    """
    Prepare data for training and testing.

    :param landmarks: Array of landmarks.
    :param labels: Array of labels.
    :param num_classes: Number of classes.
    :return: Training and testing data.
    """
    try:
        labels = tf.keras.utils.to_categorical(
            labels, num_classes=num_classes).astype(int)
        landmarks_train, landmarks_test, labels_train, labels_test = train_test_split(
            landmarks, labels, test_size=0.2)
        return landmarks_train, landmarks_test, labels_train, labels_test
    except Exception as e:
        logging.error(f"Error preparing data: {e}")
        return None, None, None, None


# Create model
def create_model(input_shape, num_classes):
    """
    Create LSTM model for gesture recognition.

    :param input_shape: Shape of the input data.
    :param num_classes: Number of classes.
    :return: Compiled LSTM model.
    """
    try:
        model = tf.keras.models.Sequential()
        model.add(tf.keras.Input(input_shape))
        model.add(LSTM(64, return_sequences=True, activation='relu'))
        model.add(LSTM(128, return_sequences=True, activation='relu'))
        model.add(LSTM(64, return_sequences=False, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        return model
    except Exception as e:
        logging.error(f"Error creating model: {e}")
        return None


# Train model
def train_model(model, landmarks_train, labels_train, landmarks_test, labels_test):
    """
    Train the LSTM model.

    :param model: The compiled model.
    :param landmarks_train: Training data.
    :param labels_train: Training labels.
    :param landmarks_test: Testing data.
    :param labels_test: Testing labels.
    :return: Training history.
    """
    try:
        log_dir = os.path.join(LOGS_PATH)
        callbacks = [
            # EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            tf.keras.callbacks.TensorBoard(log_dir=log_dir),
            # ModelCheckpoint('model.keras', save_best_only=True, monitor='val_loss'),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.00001)
        ]
        history = model.fit(landmarks_train, labels_train,
                            validation_data=(landmarks_test, labels_test),
                            epochs=300,
                            callbacks=callbacks)
        model.save(os.path.join(MODEL_PATH, MODEL_NAME))
        return history
    except Exception as e:
        logging.error(f"Error training model: {e}")
        return None


# Evaluate model
def evaluate_model(model, landmarks_test, labels_test, sign_labels):
    """
    Evaluate the trained model.

    :param model: The trained model.
    :param landmarks_test: Testing data.
    :param labels_test: Testing labels.
    :param sign_labels: Dictionary of labels.
    """
    try:
        labels_pred = model.predict(landmarks_test)
        labels_pred = np.argmax(labels_pred, axis=1)
        labels_test = np.argmax(labels_test, axis=1)
        mat = multilabel_confusion_matrix(labels_test, labels_pred)
        logging.info(f"Confusion matrix: \n{mat}")
        logging.info(f"Accuracy score of the model: {accuracy_score(labels_test, labels_pred)}")
        target_names = list(sign_labels.values())
        report = classification_report(labels_test, labels_pred, target_names=target_names)
        logging.info(f"Classification Report:\n{report}")
    except Exception as e:
        logging.error(f"Error evaluating model: {e}")


# Main Execution
def main():
    """
    Main function to load data, train and evaluate the model.
    """
    try:
        labels_dict = load_labels()
        landmarks, labels = load_dataset(labels_dict)
        if landmarks is None or labels is None:
            raise ValueError("Failed to load data.")
        num_classes = len(labels_dict)
        landmarks_train, landmarks_test, labels_train, labels_test = prepare_data(landmarks, labels, num_classes)
        if landmarks_train is None or landmarks_test is None or labels_train is None or labels_test is None:
            raise ValueError("Failed to prepare data.")
        input_shape = (NUM_FRAMES, landmarks_train.shape[2])
        model = create_model(input_shape, num_classes)
        if model is None:
            raise ValueError("Failed to create model.")
        history = train_model(model, landmarks_train, labels_train, landmarks_test, labels_test)
        if history is None:
            raise ValueError("Failed to train model.")
        evaluate_model(model, landmarks_test, labels_test, labels_dict)
    except Exception as e:
        logging.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
