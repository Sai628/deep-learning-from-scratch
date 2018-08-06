# coding=utf-8

import numpy as np


def smooth_curve(x):
    """用于平滑损失函数的图形"""
    window_len = 11
    s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
    w = np.kaiser(window_len, 2)
    y = np.convolve(w/w.sum(), s, mode='valid')
    return y[5:len(y)-5]


def shuffle_dataset(x, t):
    permutation = np.random.permutation(x.shape[0])
    if x.ndim == 2:
        x = x[permutation,:]
    else:
        x = x[permutation,:,:,:]
    t = t[permutation]

    return x, t
