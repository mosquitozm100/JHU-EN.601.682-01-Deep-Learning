import numpy as np
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

x = 0.63
w1 = 0.25
w2 = -0.11
w3 = 0.78
y = 1
a = sigmoid(w1 * x)
b = sigmoid(w2 * sigmoid(w1 * x))
y_hat = w3 *(sigmoid(w2 * sigmoid(w1 * x)))

print(a)
print(b)
print(y_hat)

print(-2 * (y - y_hat))
print(-2 * (y - y_hat) * b)
print(-2 * (y - y_hat) * w3 * b * (1 - b) * a)
print(-2 * (y - y_hat) * w3 * b * (1 - b) * w2 * a * (1 - a) * x) 
