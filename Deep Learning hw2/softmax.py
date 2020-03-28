import math
import numpy as np

W = np.array([0.3, -0.5])
X = np.array([0.2, 0.4])

def softmax(s):
    return 1 / (1 + math.exp(-s))

s = W @ X.T
print(s)
t = softmax(s)
print(t)
lambda_R = (W * W).sum() / 2
print(lambda_R)
Loss = t + lambda_R 
print(Loss)