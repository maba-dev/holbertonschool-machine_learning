#!/usr/bin/env python3
"""a function that calculates the F1 score of a confusion matrix"""


sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """define the function """
    prec = precision(confusion)
    sens = sensitivity(confusion)
    F1 = 2 * (prec * sens) / (prec + sens)
    return F1
