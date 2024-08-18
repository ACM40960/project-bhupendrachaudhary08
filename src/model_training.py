import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from config.config import ARTIFACTS_PATH, DATA_PICKLE_NAME, MODEL_NAME
from itertools import cycle


def plot_loss_accuracy(history):
    """Plot the training and validation loss and accuracy."""
    sns.set(style="whitegrid")

    epochs = range(1, len(history['train_loss']) + 1)

    # Plotting Loss
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plotting Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
    plt.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()


def plot_roc_and_pr_curves(y_test, y_score, classes):
    """Plot ROC and Precision-Recall curves."""
    # Binarize the output for One-vs-Rest
    y_test_bin = label_binarize(y_test, classes=classes)
    n_classes = len(classes)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plotting ROC Curve
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple', 'brown'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve for {classes[i]} (AUC = {roc_auc[i]:0.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Chance level (AUC = 0.5)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('One-vs-Rest Multiclass ROC')
    plt.legend(loc="lower right")

    # Compute Precision-Recall curve and PR area for each class
    precision = dict()
    recall = dict()
    pr_auc = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
        pr_auc[i] = auc(recall[i], precision[i])

    # Plotting Precision-Recall Curve
    plt.subplot(1, 2, 2)
    for i, color in zip(range(n_classes), colors):
        plt.plot(recall[i], precision[i], color=color, lw=2,
                 label=f'Precision-Recall curve for {classes[i]} (AP = {pr_auc[i]:0.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('One-vs-Rest Multiclass Precision-Recall')
    plt.legend(loc="lower left")

    plt.show()


def plot_confusion_matrix(cm, labels):
    """Plot confusion matrix."""
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()


def model_training():
    with open(os.path.join(ARTIFACTS_PATH, DATA_PICKLE_NAME), "rb") as f:
        data_dict = pickle.load(f)

    data = np.asarray(data_dict['data'])
    labels = np.asarray(data_dict['labels'])

    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

    model = RandomForestClassifier()
    # verbose = 2, n_jobs = -1

    # Track loss and accuracy for manual plotting
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }

    model.fit(x_train, y_train)

    # Generate predictions for training and validation
    y_train_pred = model.predict(x_train)
    y_val_pred = model.predict(x_test)

    # For ROC and Precision-Recall curves, we need the decision function or probability estimates
    y_score = model.predict_proba(x_test)

    # Calculate training and validation loss (log loss) and accuracy
    train_acc = accuracy_score(y_train, y_train_pred)
    print(train_acc)
    val_acc = accuracy_score(y_test, y_val_pred)
    print(val_acc)

    # Mock losses (RandomForest doesn't have a loss function like NN)
    train_loss = 1 - train_acc
    val_loss = 1 - val_acc

    # Populate history dict
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)

    # Plot loss and accuracy
    plot_loss_accuracy(history)

    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_val_pred)
    plot_confusion_matrix(cm, labels=np.unique(labels))

    # Plot ROC and Precision-Recall Curves
    plot_roc_and_pr_curves(y_test, y_score, classes=np.unique(labels))

    # Classification report
    print(classification_report(y_test, y_val_pred, target_names=np.unique(labels)))

    # Save the trained model
    os.makedirs(ARTIFACTS_PATH, exist_ok=True)
    with open(os.path.join(ARTIFACTS_PATH, MODEL_NAME), 'wb') as f:
        pickle.dump({'model': model}, f)


if __name__ == "__main__":
    model_training()
