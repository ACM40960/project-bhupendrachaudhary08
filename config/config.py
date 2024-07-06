# Import necessary packages/dependencies
import os

# Directory paths
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # Absolute path of base directory
DATASET_PATH = os.path.join(BASE_PATH, 'dataset')  # Path to the dataset directory
MODEL_PATH = os.path.join(BASE_PATH, 'models')  # Path to the models directory
LOGS_PATH = os.path.join(BASE_PATH, 'logs')  # Path to the models directory
LABELS_PATH = os.path.join(BASE_PATH, 'labels.txt')  # Path to the labels directory

# Constants for data collection
NUM_SAMPLES = 3  # Number of samples to collect for each label
NUM_FRAMES = 30  # Number of frames per sample

# Model configuration
MODEL_NAME = 'model.keras'

# Logging configuration
LOGGING_LEVEL = 'INFO'
LOGGING_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"    # Format for log messages
