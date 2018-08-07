# coding=utf-8

import os
import sys
sys.path.append(os.pardir)  # 为了导入父目录中的文件而进行的设定

import numpy as np

from deep_convnet import DeepConvNet
from dataset.mnist import load_mnist


(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

network = DeepConvNet()
network.load_params('deep_convnet_params.pkl')

sampled = 10000  # 加快速度
x_test = x_test[:sampled]
t_test = t_test[:sampled]

print('calculate accuracy (float64)...')
print(network.accuracy(x_test, t_test))

# 转换为 float16 类型
x_test = x_test.astype(np.float16)
for param in network.params.values():
    param[...] = param.astype(np.float16)

print('calculate accuracy (float16)...')
print(network.accuracy(x_test, t_test))
