# coding=utf-8

import numpy as np


def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

if __name__ == '__main__':
    for x in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        y = AND(x[0], x[1])
        print(str(x) + " -> " + str(y))

""" output result:
(0, 0) -> 0
(0, 1) -> 0
(1, 0) -> 0
(1, 1) -> 1
"""
