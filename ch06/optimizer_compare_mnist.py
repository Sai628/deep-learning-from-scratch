# coding=utf-8

import os
import sys
sys.path.append(os.pardir)  # 为了导入父目录中的文件而进行的设定

import numpy as np
import matplotlib.pyplot as plt

from dataset.mnist import load_mnist
from common.util import smooth_curve
from common.multi_layer_net import MultiLayerNet
from common.optimizer import *


# 读入数据
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# 超参数
train_size = x_train.shape[0]
batch_size = 128
max_iterations = 2000


# 设置优化算法
optimizers = {}
optimizers['SGD'] = SGD()
optimizers['Momentum'] = Momentum()
optimizers['AdaGrad'] = AdaGrad()
optimizers['Adam'] = Adam()

networks = {}
train_loss = {}
for key in optimizers.keys():
    networks[key] = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100], output_size=10)
    train_loss[key] = []


# 开始训练
for i in range(max_iterations):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    for key in optimizers.keys():
        grads = networks[key].gradient(x_batch, t_batch)
        optimizers[key].update(networks[key].params, grads)

        loss = networks[key].loss(x_batch, t_batch)
        train_loss[key].append(loss)

    if i % 100 == 0:
        print("\niteration:" + str(i))
        for key in optimizers.keys():
            loss = networks[key].loss(x_batch, t_batch)
            print(key + ": " + str(loss))


# 绘制图表
markers = {'SGD': 'o', 'Momentum': 'x', 'AdaGrad': 's', 'Adam': 'D'}
x = np.arange(max_iterations)
for key in optimizers.keys():
    plt.plot(x, smooth_curve(train_loss[key]), marker=markers[key], markevery=100, label=key)
plt.xlabel('iterations')
plt.ylabel('loss')
plt.ylim(0, 1)
plt.legend()
plt.show()
