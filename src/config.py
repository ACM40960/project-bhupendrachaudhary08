import os

# Directory paths
# Absolute path of base directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Path to the dataset directory
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
# Path to the labels directory
LABELS_FILE = os.path.join(BASE_DIR, 'labels.txt')
# Path to the directory for storing artifacts like models and data
ARTIFACTS_DIR = os.path.join(BASE_DIR, 'artifacts')

# Constants for data collection
NUM_SAMPLES = 200  # Number of samples to collect for each label

# Model configuration
MODEL_FILENAME = 'model.pkl'    # Filename for the serialized model
DATA_PICKLE_FILENAME = 'data.pickle'    # Filename for the dataset pickle file

# Logging configuration
LOGGING_LEVEL = 'INFO'
# Format for log messages
LOGGING_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
