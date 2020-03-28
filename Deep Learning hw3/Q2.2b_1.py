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
y_hat = sigmoid(w3 *(sigmoid(w2 * sigmoid(w1 * x))))

print(- y / y_hat)
print(- y / y_hat * y_hat * (1 - y_hat) * b)
print(- y / y_hat * y_hat * (1 - y_hat) * w3 * b * (1 - b) * a)
print(- y / y_hat * y_hat * (1 - y_hat) * w3 * b * (1 - b) * w2 * a * (1 - a) * x) 
