import os

# Directory paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # Absolute path of base directory
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')  # Path to the dataset directory
LABELS_FILE = os.path.join(BASE_DIR, 'labels.txt')  # Path to the labels directory
ARTIFACTS_DIR = os.path.join(BASE_DIR, 'artifacts')  # Path to the directory for storing artifacts like models and data

# Constants for data collection
NUM_SAMPLES = 50  # Number of samples to collect for each label

# Model configuration
MODEL_FILENAME = 'model.pkl'    # Filename for the serialized model
DATA_PICKLE_FILENAME = 'data.pickle'    # Filename for the dataset pickle file

# Logging configuration
LOGGING_LEVEL = 'INFO'
LOGGING_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"  # Format for log messages
