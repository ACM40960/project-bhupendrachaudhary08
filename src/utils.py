# Import necessary packages/dependencies
import logging
from typing import Dict

from config import LABELS_FILE


def load_labels() -> Dict[int, str]:
    """
    Load labels from the labels file and return them as a dictionary.

    :return: A dictionary mapping indices (int) to labels (str).
    """
    try:
        sign_labels: list[str] = []
        # Open the labels file and read all the lines
        with (open(LABELS_FILE, "r") as file):
            for line in file.readlines():
                line = line.strip()
                # Skip comments and empty lines
                if line.startswith('#') or not line.strip():
                    continue
                # Add the label to the sign_labels list
                sign_labels.append(line)
            # Return a dictionary where keys are indices and values are labels
        return {idx: label for idx, label in enumerate(sign_labels)}
    except Exception as e:
        logging.error(f"Error loading labels: {e}")
        return {}
