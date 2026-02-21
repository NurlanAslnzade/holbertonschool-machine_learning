#!/usr/bin/env python3
import numpy as np
"""ASDasdasd asdas da"""


def create_confusion_matrix(labels, logits):
    """
    Creates a confusion matrix from one-hot encoded labels and logits.
    
    Args:
        labels: numpy.ndarray of shape (m, classes) with one-hot true labels
        logits: numpy.ndarray of shape (m, classes) with one-hot predicted labels
    
    Returns:
        numpy.ndarray of shape (classes, classes) confusion matrix
    """
    y_true = np.argmax(labels, axis=1)
    y_pred = np.argmax(logits, axis=1)
    classes = labels.shape[1]
    confusion = np.zeros((classes, classes))
    for i in range(len(y_true)):
        confusion[y_true[i], y_pred[i]] += 1
    return confusion
