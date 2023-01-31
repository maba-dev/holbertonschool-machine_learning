#!/usr/bin/env python3
""" the function that calculates the weighted moving average of a data set:"""


def moving_average(data, beta):
    """calculates the weighted moving average of a data set:"""
    mv_averages = []
    avg = 0
    for i in range(len(data)):
        avg = (beta * avg) + (data[i] * (1 - beta))
        mv_averages.append(avg / (1 - beta**(i + 1)))
    return mv_averages
