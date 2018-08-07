# coding=utf-8

import os
import sys
sys.path.append(os.pardir)  # 为了导入父目录中的文件而进行的设定

import numpy as np
import pickle

from common.gradient import numerical_gradient
from common.layers import *


class DeepConvNet:
    """
    高精度的ConvNet, 识别率达到99%或更高

    网络层设定如下:

    conv - relu - conv - relu - pooling -
    conv - relu - conv - relu - pooling -
    conv - relu - conv - relu - pooling -
    affine - relu - dropout -
    affine - dropout - softmax
    """
    def __init__(self, input_dim=(1, 28, 28),
                 conv_param1={'filter_num': 16, 'filter_size': 3, 'pad': 1, 'stride': 1},
                 conv_param2={'filter_num': 16, 'filter_size': 3, 'pad': 1, 'stride': 1},
                 conv_param3={'filter_num': 32, 'filter_size': 3, 'pad': 1, 'stride': 1},
                 conv_param4={'filter_num': 32, 'filter_size': 3, 'pad': 2, 'stride': 1},
                 conv_param5={'filter_num': 64, 'filter_size': 3, 'pad': 1, 'stride': 1},
                 conv_param6={'filter_num': 64, 'filter_size': 3, 'pad': 1, 'stride': 1},
                 hidden_size=50, output_size=10):
        # 权重初始化
        # 每层中有多少个神经元连接到前层神经元(TODO: 自动计算)
        pre_node_nums = np.array([1*3*3, 16*3*3, 16*3*3, 32*3*3, 32*3*3, 64*3*3, 64*4*4, hidden_size])
        weight_init_scales = np.sqrt(2.0 / pre_node_nums)  # 使用Relu时建议的初始值(He初始值)

        self.params = {}
        pre_channel_num = input_dim[0]
        for idx, conv_param in enumerate([conv_param1, conv_param2, conv_param3, conv_param4, conv_param5, conv_param6]):
            filter_randn = np.random.randn(conv_param['filter_num'], pre_channel_num,
                                           conv_param['filter_size'], conv_param['filter_size'])
            self.params['W' + str(idx+1)] = weight_init_scales[idx] * filter_randn
            self.params['b' + str(idx+1)] = np.zeros(conv_param['filter_num'])
            pre_channel_num = conv_param['filter_num']

        self.params['W7'] = weight_init_scales[6] * np.random.randn(64*4*4, hidden_size)
        self.params['b7'] = np.zeros(hidden_size)
        self.params['W8'] = weight_init_scales[7] * np.random.randn(hidden_size, output_size)
        self.params['b8'] = np.zeros(output_size)

        # 生成层
        self.layers = []
        self.layers.append(Convolution(self.params['W1'], self.params['b1'], conv_param1['stride'], conv_param1['pad']))
        self.layers.append(Relu())
        self.layers.append(Convolution(self.params['W2'], self.params['b2'], conv_param2['stride'], conv_param2['pad']))
        self.layers.append(Relu())
        self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))

        self.layers.append(Convolution(self.params['W3'], self.params['b3'], conv_param3['stride'], conv_param3['pad']))
        self.layers.append(Relu())
        self.layers.append(Convolution(self.params['W4'], self.params['b4'], conv_param4['stride'], conv_param4['pad']))
        self.layers.append(Relu())
        self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))

        self.layers.append(Convolution(self.params['W5'], self.params['b5'], conv_param5['stride'], conv_param5['pad']))
        self.layers.append(Relu())
        self.layers.append(Convolution(self.params['W6'], self.params['b6'], conv_param6['stride'], conv_param6['pad']))
        self.layers.append(Relu())
        self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))

        self.layers.append(Affine(self.params['W7'], self.params['b7']))
        self.layers.append(Relu())
        self.layers.append(Dropout(0.5))

        self.layers.append(Affine(self.params['W8'], self.params['b8']))
        self.layers.append(Dropout(0.5))
        self.last_layer = SoftmaxWithLoss()

    def predict(self, x, train_flag=False):
        for layer in self.layers:
            if isinstance(layer, Dropout):
                x = layer.forward(x, train_flag)
            else:
                x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x, train_flag=True)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        acc = 0.0
        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx, train_flag=False)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)

        return acc / x.shape[0]

    def numerical_gradient(self, x, t):
        loss_W = lambda w: self.loss(x, t)

        grads = {}
        for idx in (1, 2, 3, 4, 5, 6):
            grads['W' + str(idx)] = numerical_gradient(loss_W, self.params['W' + str(idx)])
            grads['b' + str(idx)] = numerical_gradient(loss_W, self.params['b' + str(idx)])

        return grads

    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = self.layers.copy()
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 设定
        grads = {}
        for i, layer_idx in enumerate((0, 2, 5, 7, 10, 12, 15, 18)):
            grads['W' + str(i+1)] = self.layers[layer_idx].dW
            grads['b' + str(i+1)] = self.layers[layer_idx].db

        return grads

    def save_params(self, file_name='params.pkl'):
        params = {}
        for key, val in self.params.items():
            params[key] = val

        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name='params.pk1'):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)

        for key, val in params.items():
            self.params[key] = val

        for i, layer_idx in enumerate((0, 2, 5, 7, 10, 12, 15, 18)):
            self.layers[layer_idx].W = self.params['W' + str(i+1)]
            self.layers[layer_idx].b = self.params['b' + str(i+1)]
