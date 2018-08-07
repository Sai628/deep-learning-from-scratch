# coding=utf-8

import os
import sys
sys.path.append(os.pardir)  # 为了导入父目录中的文件而进行的设定

import numpy as np
import matplotlib.pyplot as plt

from deep_convnet import DeepConvNet
from dataset.mnist import load_mnist


(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)
network = DeepConvNet()
network.load_params('deep_convnet_params.pkl')

print('calculating test accuracy...')
# sampled = 1000  # 为了速度而减少测试数据
# x_test = x_test[:sampled]
# t_test = t_test[:sampled]

classified_ids = []  # 保存根据训练后的模型而推理得到的结果
acc = 0.0
batch_size = 100

for i in range(int(x_test.shape[0] / batch_size)):
    tx = x_test[i*batch_size:(i+1)*batch_size]
    tt = t_test[i*batch_size:(i+1)*batch_size]
    y = network.predict(tx, train_flag=False)
    y = np.argmax(y, axis=1)
    classified_ids.append(y)
    acc += np.sum(y == tt)

acc = acc / x_test.shape[0]
print('test accuracy: ' + str(acc))

classified_ids = np.array(classified_ids)
classified_ids = classified_ids.flatten()

fig = plt.figure()
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.2, wspace=0.2)

max_view = 20
current_view = 1
mis_pairs = {}
for i, val in enumerate(classified_ids == t_test):
    if not val:  # 获取推理结果与真实标签不符的数据
        ax = fig.add_subplot(4, 5, current_view, xticks=[], yticks=[])
        ax.imshow(x_test[i].reshape(28, 28), cmap=plt.cm.gray_r, interpolation='nearest')
        mis_pairs[current_view] = (t_test[i], classified_ids[i])

        current_view += 1
        if current_view > max_view:
            break

print('misclassified result:')
print('{view index: (label, inference), ...}')
print(mis_pairs)

plt.show()
