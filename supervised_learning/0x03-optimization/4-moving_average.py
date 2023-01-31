#!/usr/bin/env python3
""" the function that calculates the weighted moving average of a data set:"""


def moving_average(data, beta):
    """calculates the weighted moving average of a data set:"""
    mv_averages = []
    avg = 0
    bias_correction = 1
    for i, value in enumerate(data):
        avg = beta * avg + (1 - beta) * value
        bias_correction = bias_correction * beta
        if i + 1 == len(data):
            mv_averages.append(avg / (1 - bias_correction))
        else:
            mv_averages.append(avg / (1 - bias_correction))
    return mv_averages
