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


def conv_output_size(input_size, filter_size, stride=1, pad=0):
    return (input_size + 2 * pad - filter_size) / stride + 1


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """
    将输入数据展开以适合滤波器

    :param input_data: 由(数据量, 通道, 高, 长)的4维数据构成的输入数据
    :param filter_h: 滤波器的高
    :param filter_w: 滤波器的长
    :param stride: 步幅
    :param pad: 填充
    :return: 展开后的二维数据
    """

    N, C, H, W = input_data.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """
    im2col 的逆处理

    :param col: 待处理的二维数据
    :param input_shape: 输入数据的形状. 如: (10, 1, 28, 28)
    :param filter_h: 滤波器的高
    :param filter_w: 滤波器的长
    :param stride: 步幅
    :param pad: 填充
    :return:
    """

    N, C, H, W = input_shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad: W + pad]