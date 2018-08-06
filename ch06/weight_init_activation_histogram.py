# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def tanh(x):
    return np.tanh(x)


x = np.random.randn(1000, 100)  # 1000个数据
node_num = 100  # 各隐藏层的节点(神经元)数
hidden_layer_size = 5  # 隐藏层有5层
activations = {}  # 激活值的结果保存在这里


for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i-1]

    # 尝试不同的权重初始化
    w = np.random.randn(node_num, node_num) * 1
    # w = np.random.randn(node_num, node_num) * 0.01
    # w = np.random.randn(node_num, node_num) * np.sqrt(1.0 / node_num)  # "Xavier初始值". 激活函数为Sigmoid或tanh时, 推荐使用
    # w = np.random.randn(node_num, node_num) * np.sqrt(2.0 / node_num)  # "He初始值". 激活函数为ReLU时, 推荐使用

    z = np.dot(x, w)
    a = sigmoid(z)
    # a = relu(z)
    # a = tanh(z)

    activations[i] = a

# 绘制直方图
for i, a in activations.items():
    plt.subplot(1, len(activations), i+1)
    plt.title(str(i+1) + '-layer')
    if i != 0:
        plt.yticks([], [])
    plt.hist(a.flatten(), 30, range=(0, 1))

plt.show()
