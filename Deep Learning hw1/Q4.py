import math
import numpy as np

W = np.array([[0.01,-0.05,0.1,0.05],
            [0.7,0.2,0.05,0.16],
            [0.0,-0.45,-0.2,0.03]])
x = np.array([-15, 22, -44, 56]).T
b = np.array([0.0, 0.2, -0.3]).T
y = np.array([0, 0, 1]).T

def calc_f(W, x, b):
    return np.dot(W, x) + b

def calc_hinge_loss(f, y):
    total_loss = 0
    yi = np.where(y == 1)
    #print(type(yi))
    for j in range(0, f.shape[0]):
        #print(f[j] - f[yi] + 1)
        if j != yi:
            total_loss += max(0, f[j] - f[yi] + 1)
    return total_loss

def calc_softmax_loss(f, y):
    total_loss = 0
    yi = np.where(y == 1)
    total_loss = -np.log(np.exp(f[yi]) / np.sum(np.exp(f)))
    return total_loss

if __name__ == "__main__":
    f = calc_f(W, x, b)
    print(f)
    print('hinge_loss：', calc_hinge_loss(f, y))
    print('softmax_loss:', calc_softmax_loss(f,y))

