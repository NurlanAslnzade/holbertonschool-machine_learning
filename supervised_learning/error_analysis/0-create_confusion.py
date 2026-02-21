#!/usr/bin/env python3
"""Module that contains the function create_confusion_matrix"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Returning confusion matrix
    """
    return np.matmul(labels.T, logits)
