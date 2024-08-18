# Import necessary packages/dependencies
import os

# Directory paths
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # Absolute path of base directory
DATASET_PATH = os.path.join(BASE_PATH, 'dataset')  # Path to the dataset directory
LABELS_PATH = os.path.join(BASE_PATH, 'labels.txt')  # Path to the labels directory
ARTIFACTS_PATH = os.path.join(BASE_PATH, 'artifacts')  # New path for storing pickle files and models

# Constants for data collection
NUM_SAMPLES = 50  # Number of samples to collect for each label

# Model configuration
# MODEL_NAME = 'model.keras'
MODEL_NAME = 'model.pkl'
DATA_PICKLE_NAME = 'data.pickle'

# Logging configuration
LOGGING_LEVEL = 'INFO'
LOGGING_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"  # Format for log messages
