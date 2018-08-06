# coding=utf-8

import os
import sys
sys.path.append(os.pardir)  # 为了导入父目录中的文件而进行的设定
from collections import OrderedDict

import numpy as np

from common.layers import *
from common.gradient import numerical_gradient


class MultiLayerNet:
    def __init__(self, input_size, hidden_size_list, output_size,
                 activation='relu', weight_init_std='relu', weight_decay_lambda=0):
        """
        具有全连接的多层神经网络

        :param input_size: 输入层大小(MNIST中为784)
        :param hidden_size_list: 隐藏层中神经元数量的列表(e.g. [100, 100, 100])
        :param output_size: 输出层大小(MNIST中为10)
        :param activation: 激活函数. 'relu' 或 'sigmoid'
        :param weight_init_std: 指定权重的标准偏差(e.g. 0.01)
            若该参数为 'relu' 或 'he' 时, 使用Relu函数的初始化值;
            若该参数为 'sigmoid' 或 'xavier' 时, 使用Sigmoid函数的初始化值.
        :param weight_decay_lambda: 权值衰减参数(L2范数), 用于控制正则化强度
        """
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)
        self.weight_decay_lambda = weight_decay_lambda
        self.params = {}

        # 初始化权重
        self.__init_weight(weight_init_std)

        # 生成层
        activation_layer = {'sigmoid': Sigmoid, 'relu': Relu}
        self.layers = OrderedDict()
        for idx in range(1, self.hidden_layer_num + 1):
            self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)], self.params['b' + str(idx)])
            self.layers['Activation_function' + str(idx)] = activation_layer[activation]()

        idx = self.hidden_layer_num + 1
        self.layers['Affine' + str(idx)] = Affine(self.params['W' + str(idx)], self.params['b' + str(idx)])
        self.last_layer = SoftmaxWithLoss()

    def __init_weight(self, weight_init_std):
        """
        设置初始化权重值

        :param weight_init_std: 指定权重的标准偏差(e.g. 0.01)
            若该参数为 'relu' 或 'he' 时, 使用 "He初始值";
            若该参数为 'sigmoid' 或 'xavier' 时, 使用 "Xavier初始值".
        :return:
        """
        all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]
        for idx in range(1, len(all_size_list)):
            scale = weight_init_std
            if str(weight_init_std).lower() in ('relu', 'he'):
                scale = np.sqrt(2.0 / all_size_list[idx - 1])  # Relu 使用推荐的初始值
            elif str(weight_init_std).lower() in ('sigmoid', 'xavier'):
                scale = np.sqrt(1.0 / all_size_list[idx - 1])  # Sigmoid 使用推荐的初始值

            self.params['W' + str(idx)] = scale * np.random.randn(all_size_list[idx - 1], all_size_list[idx])
            self.params['b' + str(idx)] = np.zeros(all_size_list[idx])

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        y = self.predict(x)

        weight_decay = 0
        for idx in range(1, self.hidden_layer_num + 2):
            W = self.params['W' + str(idx)]
            # L2范数的权值衰减为: (1/2)*λ*(W^2)
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W ** 2)

        return self.last_layer.forward(y, t) + weight_decay

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        """
        基于数值微分法计算梯度

        :param x: 输入数据
        :param t: 监督数据
        :return: 以字典形式返回各层的权重与偏置值
            grads['W1'], grads['W2']... 各层的权重值
            grads['b1], grads['b2]... 各层的偏置值
        """
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        for idx in range(1, self.hidden_layer_num + 2):
            grads['W' + str(idx)] = numerical_gradient(loss_W, self.params['W' + str(idx)])
            grads['b' + str(idx)] = numerical_gradient(loss_W, self.params['b' + str(idx)])

        return grads

    def gradient(self, x, t):
        """
        基于误差反向传播法计算梯度

        :param x: 输入数据
        :param t: 监督数据
        :return: 以字典形式返回各层的权重与偏置值
            grads['W1'], grads['W2']... 各层的权重值
            grads['b1], grads['b2]... 各层的偏置值
        """
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 设定
        grads = {}
        for idx in range(1, self.hidden_layer_num + 2):
            # 对于所有权重, 权值衰减方法都会为损失函数加上 (1/2)*λ*(W^2),
            # 因此, 在求权重梯度的计算中, 要为之前的误差反向传播法的结果加上正则化项的导数 λ*W
            grads['W' + str(idx)] = self.layers['Affine' + str(idx)].dW + \
                                    self.weight_decay_lambda * self.layers['Affine' + str(idx)].W
            grads['b' + str(idx)] = self.layers['Affine' + str(idx)].db

        return grads
