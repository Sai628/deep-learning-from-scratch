# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt


def numerical_diff(f, x):
    h = 1e-4  # 0.0001
    return (f(x+h) - f(x-h)) / (2 * h)


def function_1(x):
    return 0.01 * (x**2) + 0.1 * x


def tangent_line(f, x):
    d = numerical_diff(f, x)
    print(d)
    y = f(x) - d * x
    return lambda t: d * t + y


x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")

tf_5 = tangent_line(function_1, 5)  # x=5 处的切线
y2 = tf_5(x)

tf_10 = tangent_line(function_1, 10)  # x=10 处的切线
y3 = tf_10(x)

plt.plot(x, y)
plt.plot(x, y2)
plt.plot(x, y3)
plt.show()
