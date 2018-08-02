# coding=utf-8

from layer_naive import *


apple = 100
apple_num = 2
tax = 1.1

# layer
mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()

# forward
apple_price = mul_apple_layer.forward(apple, apple_num)
price = mul_tax_layer.forward(apple_price, tax)

# backward
dprice = 1
dapple_price, dtax = mul_tax_layer.backward(dprice)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)

print("price:", int(price))  # price: 220
print("dApple:", dapple)  # dApple: 2.2
print("dApple_num:", int(dapple_num))  # dApple_num: 110
print("dTax:", dtax)  # dTax: 200
