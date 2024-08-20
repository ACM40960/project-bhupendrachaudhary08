from itertools import cycle
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
import pickle

from config import ARTIFACTS_DIR, DATA_PICKLE_FILENAME, MODEL_FILENAME, LOGGING_FORMAT, LOGGING_LEVEL

# Configure logging settings
logging.basicConfig(level=LOGGING_LEVEL, format=LOGGING_FORMAT)


def plot_accuracy_bar(train_acc: float, val_acc: float) -> None:
    """
    Plot the training and validation accuracy as a bar chart.

    :param train_acc: Training accuracy as a float.
    :param val_acc: Validation accuracy as a float.
    :return: None
    """
    sns.set(style="whitegrid")

    plt.figure(figsize=(8, 5))
    bar_width = 0.35
    index = np.arange(1)
    plt.bar(index, train_acc, bar_width, label='Training Accuracy', color='b')
    plt.bar(index + bar_width, val_acc, bar_width, label='Validation Accuracy', color='r')

    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xticks([])
    plt.legend()

    plt.show()


def plot_roc_and_pr_curves(y_test: np.ndarray, y_score: np.ndarray, classes: np.ndarray, max_display: int = 10) -> None:
    """
    Plot ROC and Precision-Recall curves, limiting the number of classes displayed in the legend.

    :param y_test: Array of true labels (np.ndarray).
    :param y_score: Array of predicted probabilities (np.ndarray).
    :param classes: Array of class labels (np.ndarray).
    :param max_display: Maximum number of classes to display in the legend (default is 10).
    :return: None
    """
    y_test_bin = label_binarize(y_test, classes=classes)
    n_classes = len(classes)

    fpr = {}
    tpr = {}
    roc_auc = {}

    precision = {}
    recall = {}
    pr_auc = {}

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
        pr_auc[i] = auc(recall[i], precision[i])

    # Sort classes by ROC AUC for displaying top N
    top_classes = sorted(range(n_classes), key=lambda j: roc_auc[j], reverse=True)[:max_display]

    # Plot ROC Curve
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    # Get colors from the 'tab20' colormap
    colormap = plt.get_cmap('tab20')
    colors = cycle(colormap(i) for i in range(20))  # Extract colors from the colormap
    for i, color in zip(top_classes, colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'ROC curve for {classes[i]} (AUC = {roc_auc[i]:0.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Chance level (AUC = 0.5)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Top Class ROC Curves')
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize='small')

    # Plot Precision-Recall Curve
    plt.subplot(1, 2, 2)
    for i, color in zip(top_classes, colors):
        plt.plot(recall[i], precision[i], color=color, lw=2, label=f'PR curve for {classes[i]} (AP = {pr_auc[i]:0.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Top Class Precision-Recall Curves')
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize='small')

    plt.tight_layout()
    plt.show()

    # Log detailed class info
    logging.info("Detailed AUC and AP scores for all classes:")
    for i in range(n_classes):
        logging.info(f'Class {classes[i]} - AUC: {roc_auc[i]:.2f}, AP: {pr_auc[i]:.2f}')


def plot_confusion_matrix(cm: np.ndarray, labels: np.ndarray) -> None:
    """
    Plot the confusion matrix using a heatmap.

    :param cm: Confusion matrix as a numpy array.
    :param labels: Array of label names (np.ndarray).
    :return: None
    """
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()


def train_model() -> None:
    """
    Train a RandomForest model on the extracted dataset, evaluate it, and save the trained model.

    :return: None
    """
    try:
        with open(os.path.join(ARTIFACTS_DIR, DATA_PICKLE_FILENAME), "rb") as f:
            data_dict = pickle.load(f)

        data = np.asarray(data_dict['data'])
        labels = np.asarray(data_dict['labels'])

        # Split the dataset into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

        model = RandomForestClassifier()

        # Train the model
        model.fit(x_train, y_train)

        # Generate predictions for training and validation
        y_train_pred = model.predict(x_train)
        y_test_pred = model.predict(x_test)

        # For ROC and Precision-Recall curves
        y_score = model.predict_proba(x_test)

        # Calculate training and validation accuracy
        train_acc = accuracy_score(y_train, y_train_pred)
        val_acc = accuracy_score(y_test, y_test_pred)

        logging.info(f"Training accuracy: {train_acc:.4f}")
        logging.info(f"Validation accuracy: {val_acc:.4f}")

        # Plot accuracy bar chart
        plot_accuracy_bar(train_acc, val_acc)

        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_test_pred)
        plot_confusion_matrix(cm, labels=np.unique(labels))

        # Plot ROC and Precision-Recall Curves
        plot_roc_and_pr_curves(y_test, y_score, classes=np.unique(labels))

        # Print classification report
        logging.info("Classification Report:")
        logging.info("\n" + classification_report(y_test, y_test_pred, target_names=np.unique(labels)))

        # Save the trained model
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        with open(os.path.join(ARTIFACTS_DIR, MODEL_FILENAME), 'wb') as f:
            pickle.dump({'model': model}, f)
        logging.info(f"Model saved to {os.path.join(ARTIFACTS_DIR, MODEL_FILENAME)}")
    except Exception as e:
        logging.error(f"An error occurred during model training: {e}")


if __name__ == "__main__":
    train_model()
