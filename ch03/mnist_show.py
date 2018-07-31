# coding=utf-8

import os
import sys
sys.path.append(os.pardir)  # 为了导入父目录中的文件而进行的设定

import numpy as np
from dataset.mnist import load_mnist
from PIL import Image


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=False, flatten=True)

img = x_train[0]
lable = t_train[0]
print(lable)  # 5

print(img.shape)  # (784,)
img = img.reshape(28, 28)  # 把图像的形状变成原来的样子
print(img.shape)  # (28, 28)

img_show(img)
