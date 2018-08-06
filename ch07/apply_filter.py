# coding=utf-8

import os
import sys
sys.path.append(os.pardir)  # 为了导入父目录中的文件而进行的设定

import numpy as np
import matplotlib.pyplot as plt

from simple_convnet import SimpleConvNet
from matplotlib.image import imread
from common.layers import Convolution


def filter_show(filters, nx=4, show_num=16):
    fig = plt.figure()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    for i in range(show_num):
        ax = fig.add_subplot(4, 4, i+1, xticks=[], yticks=[])
        ax.imshow(filters[i, 0], cmap=plt.cm.gray_r, interpolation='nearest')


network = SimpleConvNet(input_dim=(1, 28, 28),
                        conv_param={'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                        hidden_size=100, output_size=10, weight_init_std=0.01)

# 学习后的权重
network.load_params('params.pkl')
filter_show(network.params['W1'], 16)

img = imread('../dataset/lena_gray.png')
img = img.reshape(1, 1, *img.shape)

fig = plt.figure()
for i in range(16):
    w = network.params['W1'][i]
    b = 0  # b = network.params['b1'][i]

    w = w.reshape(1, *w.shape)
    conv_layer = Convolution(w, b)
    out = conv_layer.forward(img)
    out = out.reshape(out.shape[2], out.shape[3])

    ax = fig.add_subplot(4, 4, i+1, xticks=[], yticks=[])
    ax.imshow(out, cmap=plt.cm.gray_r, interpolation='nearest')

plt.show()
