# Import necessary packages/dependencies
import cv2
import logging
import os
from typing import Tuple, Dict, Any

from config.config import LABELS_PATH


def load_labels() -> Dict[int, str]:
    """
    Load labels from the labels file.
    :return: Dictionary mapping indices to labels.
    """
    try:
        sign_labels: list[str] = []
        # Open the labels file and read all the lines
        with (open(LABELS_PATH, "r") as file):
            # sign_labels = {
            #     idx: line.strip()
            #     for idx, line in enumerate(file)
            #     if not line.startswith('#') and line.strip()
            # }
            for line in file.readlines():
                line = line.strip()
                # Skip lines that are comments
                if line.startswith('#') or not line.strip():
                    continue
                # Add the label to the sign_labels list
                sign_labels.append(line)
            # Create a dictionary mapping indices to labels
        return {idx: label for idx, label in enumerate(sign_labels)}
        # return sign_labels
    except Exception as e:
        # Log an error if an exception occurs
        logging.error(f"Error loading labels: {e}")
        return {}
