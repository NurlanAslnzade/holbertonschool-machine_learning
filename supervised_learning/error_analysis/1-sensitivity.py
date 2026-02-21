#!/usr/bin/env python3
"""Script to calculate the sensitivity in a
    confusion matrix
"""

import numpy as np


def sensitivity(confusion):
    """
    finding sensivity
    """
    TP = np.diag(confusion)
    FN = np.sum(confusion, axis=1) - TP
    TPR = TP / (TP + FN)
    return TPR
