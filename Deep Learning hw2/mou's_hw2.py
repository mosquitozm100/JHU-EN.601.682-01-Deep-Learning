# -*- coding: utf-8 -*-
"""Mou's HW2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1dCDrKFfaKFByu27RgRSVKywia-TmimgV

# Q1: Scalar Computation Graph and Gradient

## TODO
"""

import math
def g_w1(x1: float, x2: float, w1: float, w2: float) -> float:
    g = x1 * math.exp(-(w1 * x1 + w2 * x2)) / ((1 + math.exp(-(w1 * x1 + w2 * x2))) ** 2) + w1
    return g

def g_w2(x1: float, x2: float, w1: float, w2: float) -> float:
    g = x2 * math.exp(-(w1 * x1 + w2 * x2)) / ((1 + math.exp(-(w1 * x1 + w2 * x2))) ** 2) + w2
    return g

def g_x1(x1: float, x2: float, w1: float, w2: float) -> float:
    g = w1 * math.exp(-(w1 * x1 + w2 * x2)) / ((1 + math.exp(-(w1 * x1 + w2 * x2))) ** 2)
    return g

def g_x2(x1: float, x2: float, w1: float, w2: float) -> float:
    g = w2 * math.exp(-(w1 * x1 + w2 * x2)) / ((1 + math.exp(-(w1 * x1 + w2 * x2))) ** 2)
    return g

x1 = 0.2
x2 = 0.4
w1 = 0.3
w2 = -0.5

grads_x1 = []
grads_x2 = []
grads_w1 = []
grads_w2 = []
w1s = [w1]
w2s = [w2]

for i in range(30):
    grad_w1 = g_w1(x1, x2, w1, w2)    
    grad_w2 = g_w2(x1, x2, w1, w2)
    
    w2 += -0.01 * grad_w2
    w1 += -0.01 * grad_w1
    
    w1s.append(w1)
    w2s.append(w2)

    grads_w1.append(grad_w1)
    grads_w2.append(grad_w2)

import matplotlib.pyplot as plt
t = range(30)
plt.plot(t, grads_w1, label = 'grads_w1')
plt.plot(t, grads_w2, label = 'grads_w2')
plt.legend()

"""# Q3: Neural Network From Scratch

In this section, we will implement a neural network to solve MNIST, a hand-written digits (0-9) classification dataset.

## Set up
"""

import numpy as np
import matplotlib.pyplot as plt

from typing import Dict
from functools import partial

from tqdm import tqdm
tqdm.monitor_interval = 0
tqdm = partial(tqdm, bar_format='{l_bar}{r_bar}')

import torch
import torch.nn.functional as F
import torchvision

TRAIN_SIZE = 50_000

train = torchvision.datasets.MNIST('./data', train=True, transform=None, target_transform=None, download=True)
test = torchvision.datasets.MNIST('./data', train=False, transform=None, target_transform=None, download=True)

train_x = train.data.float().numpy()
train_y = train.targets.numpy()

shuffle_idx = np.arange(len(train_x))
np.random.RandomState(0).shuffle(shuffle_idx)
train_x = train_x[shuffle_idx]
train_y = train_y[shuffle_idx]

dev_x, dev_y = train_x[TRAIN_SIZE:], train_y[TRAIN_SIZE:]
train_x, train_y = train_x[:TRAIN_SIZE], train_y[:TRAIN_SIZE]

test_x = test.data.float().numpy()
test_y = test.targets.numpy()

"""Sample of images"""

i = 4048 #@param {type: "slider", min: 0, max: 10000}
print(f'label = {train_y[i]}')
plt.imshow(train_x[i])
plt.show()

"""Each images have the same shape of 28-by-28. We flatten the matrix and pretend it is just one long vector."""

NB_FEAT = 28 * 28

"""Normalize the feature. Note that we only use the training set to compute the mean and standard derivation."""

mean = train_x.mean()
std = train_x.std()

train_x = (train_x - mean) / (std + 1e-7)
dev_x = (dev_x - mean) / (std + 1e-7)
test_x = (test_x - mean) / (std + 1e-7)

train_x = train_x.reshape(-1, NB_FEAT)
dev_x = dev_x.reshape(-1, NB_FEAT)
test_x = test_x.reshape(-1, NB_FEAT)

def row_logsumexp(x):
    # numerical stablization
    x_max = x.max(axis=1).reshape(-1, 1)
    return x_max + np.log(np.exp(x - x_max).sum(axis=1)).reshape(-1, 1)

class Parameters:
    def __init__(self):
        self.param: Dict[str, np.ndarray] = {}
        self.grad: Dict[str, np.ndarray] = {}

    def set_param(self, key: str, param: np.ndarray):
        self.param[key] = param
        self.grad[key] = np.zeros_like(param)

    def get_param(self, key):
        assert key in self.param, f'variable {key} is not part of the Parameter'
        return self.param[key]

    def accumlate_grad(self, key: str, grad: np.ndarray):
        assert key in self.param, f'variable {key} is not part of the Parameter'
        assert self.param[key].shape == grad.shape, f'for variable {key}, the shape of parameter and the shape of gradient is not matched'
        self.grad[key] += grad

    def zero_grad(self):
        for key in self.param:
            self.grad[key] = np.zeros_like(self.param[key])

    def apply_grad(self, lr: float):
        for key in self.param.keys():
            assert self.param[key].shape == self.grad[key].shape, f'for variable {key}, the shape of parameter and the shape of gradient is not matched'
            self.param[key] -= self.grad[key] * lr

def init_linear(input_dim, output_dim):
    return np.random.RandomState(0).randn(input_dim, output_dim) * np.sqrt(2 / input_dim)

def main_training(params: Parameters, forward_and_backward, train_x, train_y, dev_x, dev_y, batch_size, learning_rate, nb_epochs):
    assert isinstance(params, Parameters)
    step = 0
    best_acc = 0
    record_dev_acc = []
    for epoch in tqdm(range(nb_epochs)):
        train_loss = []
        lr = learning_rate
        for i in range(0, len(train_x), batch_size):
            batch_x = train_x[i:i+batch_size]
            batch_y = train_y[i:i+batch_size]

            loss, probs = forward_and_backward(batch_x, batch_y, params)
            params.apply_grad(lr)

            train_loss.append(loss)
            step += 1

        accs = []
        for i in range(0, len(dev_x), batch_size):
            batch_x = dev_x[i:i+batch_size]
            batch_y = dev_y[i:i+batch_size]

            loss, probs = forward_and_backward(batch_x, batch_y, params, grad=False)
            pred_y = probs.argmax(axis=1)
            accs.append(pred_y == batch_y)
        acc = np.mean(accs) * 100
        record_dev_acc.append(acc)
        print(f'epoch {epoch} train loss = {np.mean(train_loss):.3f} dev accuracy = {acc:.2f}%')
        if acc > best_acc:
            best_acc = acc

    print(f'best dev accuracy = {best_acc:.2f}%')
    return record_dev_acc

"""## Q2.1 Linear Classifier

Let $x \in R^{1\times N}$ be a single sample. $N$ is the number of the feature dimension and $C$ is the number of class in the final classification. Note $N=28*28$ and $C=10$

In the softmax layer, $W_{sm} \in R^{N \times C}$. The unnormalized score $s \in R^{C}$ is

$$s = x\cdot W_{sm}$$

With softmax function, we get the probability $p(c_i)$ of class $i$ where $i \in \{1,\cdots,C\}$

$$p(c_i) = \frac {\exp(s_i)} {\sum_i \exp(s_i)}$$

where $s_i$ is the $i^{th}$ element of $s$.

Assuming $y \in \{1,\cdots,C\}$, we want to minimize $-\log p(c_y)$, the cross-entropy loss between one hot true distribution `q(c_i) = 1 if i == y else 0` and $p(c_i)$.

We will take the average $-\log p(c_y)$ of a batch of x as the loss.

In `linear_classifier_forward_and_backward`, `batch_x` and `batch_y` are matrixs, where each row are $x$ and $y$, respectively.
"""

linear_classifier = Parameters()
linear_classifier.set_param('w_sm', init_linear(NB_FEAT, 10))

"""### TODO"""

def linear_classifier_forward_and_backward(batch_x, batch_y, param: Parameters, grad=True) -> (float, np.ndarray):
    '''
    compute the loss and the gradient of each parameter
    return loss (float) and probs p(c_i) (matrix with shape batch_size x C)
    '''
    param.zero_grad()
    w_sm = param.get_param('w_sm')

    #print(batch_x.shape)
    #print(batch_y.shape)
    #print(w_sm.shape)
    # compute loss and probs
    batch_s = batch_x @ w_sm
    #probs = np.zeros((len(batch_x), 10))
    logprobs = batch_s - row_logsumexp(batch_s)
    loss = -logprobs[np.arange(len(batch_y)), batch_y].mean()
    probs = np.exp(logprobs)
    #probs = np.exp(batch_s) / np.exp(batch_s).sum(axis = 1).reshape(-1, 1)
    #loss = - np.log(probs[range(len(batch_y)),batch_y]).mean()
    #raise NotImplementedError
    #print(probs.shape)

    if not grad:
        return loss, probs

    # compute gradient
    #w_sm_grad = np.zeros_like(w_sm)
    #probs = np.exp(batch_s) / np.exp(batch_s).sum(axis = 1).reshape(-1, 1)
    probs_copy = np.copy(probs)
    probs[np.arange(len(batch_y)),batch_y] -= 1
    w_sm_grad = batch_x.T @ probs / len(batch_y)
    #print(w_sm_grad.shape)
    #raise NotImplementedError
    probs = np.copy(probs_copy)

    # save the gradient
    param.accumlate_grad('w_sm', w_sm_grad)
    return loss, probs

"""### Loss & Gradient Check

If all assertions are passed, it means you are good to go.
"""

# use a prime number for batch size would make debugging easier
bs = 7
batch_x = train_x[:bs]
batch_y = train_y[:bs]
loss, _ = linear_classifier_forward_and_backward(batch_x, batch_y, linear_classifier)

w_sm = linear_classifier.get_param('w_sm')
_w_sm = torch.tensor(w_sm, dtype=torch.double, requires_grad=True)
_batch_x = torch.tensor(batch_x, dtype=torch.double, requires_grad=False)
_batch_y = torch.tensor(batch_y, dtype=torch.long, requires_grad=False)

_logits = F.linear(_batch_x, _w_sm.T)
_logprobs = F.log_softmax(_logits, dim=-1)
_loss = -_logprobs[torch.arange(len(_batch_y)), _batch_y].mean()
_loss.backward()

assert np.isclose(loss, _loss.item())
#print(_w_sm.grad.numpy())
#print(linear_classifier.grad['w_sm'])
assert np.allclose(_w_sm.grad.numpy(), linear_classifier.grad['w_sm'])

"""### Training"""

BS = 50
LR = 0.005
NB_EPOCH = 20

linear_classifier = Parameters()
linear_classifier.set_param('w_sm', init_linear(NB_FEAT, 10))

one_linear_classifier_dev_loss = main_training(linear_classifier, linear_classifier_forward_and_backward, train_x, train_y, dev_x, dev_y, BS, LR, NB_EPOCH)

"""## Q2.1 MLP with Single Hidden Layer

Let $x \in R^{1\times N}$ be a single sample. $N$ is the number of the feature dimension and $C$ is the number of class in the final classification.

For the first layer, $W_1 \in R^{N\times \frac{N}{2}}$ and $b_1 \in R^{\frac{N}{2}}$

$$\bar{h}_1 = x\cdot W_1 + b_1$$

$$h_1 = ReLU(\bar{h}_1)$$

In the softmax layer, $W_{sm} \in R^{\frac{N}{2} \times C}$. The unnormalized score $s \in R^{C}$ is

$$s = h_1\cdot W_{sm}$$

With softmax function, we get the probability $p(c_i)$ of class $i$ where $i \in \{1,\cdots,C\}$

$$p(c_i) = \frac {\exp(s_i)} {\sum_i \exp(s_i)}$$

where $s_i$ is the $i^{th}$ element of $s$.

Assuming $y \in \{1,\cdots,C\}$, we want to minimize $-\log p(c_y)$, the cross-entropy loss between one hot true distribution `q(c_i) = 1 if i == y else 0` and $p(c_i)$.

We will take the average $-\log p(c_y)$ of a batch of x as the loss.

In `mlp_single_hidden_forward_and_backward`, `batch_x` and `batch_y` are matrixs, where each row are $x$ and $y$, respectively.
"""

mlp_single_hidden = Parameters()
mlp_single_hidden.set_param('w1', init_linear(NB_FEAT, NB_FEAT // 2))
mlp_single_hidden.set_param('b1', np.zeros(NB_FEAT // 2))
mlp_single_hidden.set_param('w_sm', init_linear(NB_FEAT // 2, 10))

"""### TODO"""

def relu(x):
    return np.maximum(0, x)

def mlp_single_hidden_forward_and_backward(batch_x, batch_y, param: Parameters, grad=True):
    '''
    compute the loss and the gradient of each parameter
    return loss (float) and p(c_i) (matrix with shape batch_size x C)
    '''
    param.zero_grad()
    w1 = param.get_param('w1')
    b1 = param.get_param('b1')
    w_sm = param.get_param('w_sm')

    #print(batch_x.shape)
    #print(batch_y.shape)
    #print(w1.shape)
    #print(b1.shape)
    #print(w_sm.shape)

    # compute loss and probs
    h1_bar = batch_x @ w1 + b1.reshape(1, -1)
    h1 = relu(h1_bar)
    #print(h1.shape)
    batch_s = h1 @ w_sm
    logprobs = batch_s - row_logsumexp(batch_s)
    probs = np.exp(logprobs)
    loss = -logprobs[np.arange(len(batch_y)), batch_y].mean()
    #raise NotImplementedError

    if not grad:
        return loss, probs

    # compute gradient
    w1_grad = np.zeros_like(w1)
    b1_grad = np.zeros_like(b1)
    w_sm_grad = np.zeros_like(w_sm)
    #print(probs.shape)
    probs_copy = np.copy(probs)
    probs[np.arange(len(batch_y)),batch_y] -= 1
    probs /= len(batch_y)
    w_sm_grad = h1.T @ probs
    h1_grad = probs @ w_sm.T
    index_of_h1_bar_smaller_than_0 = np.where(h1_bar < 0)
    #print(index_of_h1_bar_smaller_than_0)
    h1_grad[index_of_h1_bar_smaller_than_0] = 0
    w1_grad = batch_x.T @ h1_grad
    b1_grad = h1_grad.sum(axis = 0)
    #raise NotImplementedError
    #print(h1_grad)
    probs = np.copy(probs_copy)

    param.accumlate_grad('w1', w1_grad)
    param.accumlate_grad('b1', b1_grad)
    param.accumlate_grad('w_sm', w_sm_grad)
    return loss, probs

"""### Loss & Gradient Check

If all assertions are passed, it means you are good to go.
"""

# use a prime number for batch size would make debugging easier
bs = 7
batch_x = train_x[:bs]
batch_y = train_y[:bs]
loss, _ = mlp_single_hidden_forward_and_backward(batch_x, batch_y, mlp_single_hidden)

w1 = mlp_single_hidden.get_param('w1')
b1 = mlp_single_hidden.get_param('b1')
w_sm = mlp_single_hidden.get_param('w_sm')
_w1 = torch.tensor(w1, dtype=torch.double, requires_grad=True)
_b1 = torch.tensor(b1, dtype=torch.double, requires_grad=True)
_w_sm = torch.tensor(w_sm, dtype=torch.double, requires_grad=True)
_batch_x = torch.tensor(batch_x, dtype=torch.double, requires_grad=False)
_batch_y = torch.tensor(batch_y, dtype=torch.long, requires_grad=False)

_h1 = F.linear(_batch_x, _w1.T) + _b1
_h1 = F.relu(_h1)
_logits = F.linear(_h1, _w_sm.T)
_logprobs = F.log_softmax(_logits, dim=-1)
_loss = -_logprobs[torch.arange(len(_batch_y)), _batch_y].mean()
_loss.backward()

assert np.isclose(loss, _loss.item())
assert np.allclose(_w_sm.grad.numpy(), mlp_single_hidden.grad['w_sm'])
assert np.allclose(_w1.grad.numpy(), mlp_single_hidden.grad['w1'])
assert np.allclose(_b1.grad.numpy(), mlp_single_hidden.grad['b1'])

"""### Train!"""

BS = 50
LR = 0.005
NB_EPOCH = 20

mlp_single_hidden = Parameters()
mlp_single_hidden.set_param('w1', init_linear(NB_FEAT, NB_FEAT // 2))
mlp_single_hidden.set_param('b1', np.zeros(NB_FEAT // 2))
mlp_single_hidden.set_param('w_sm', init_linear(NB_FEAT // 2, 10))

MLP_with_single_hidden_layer_dev_loss = main_training(mlp_single_hidden, mlp_single_hidden_forward_and_backward, train_x, train_y, dev_x, dev_y, BS, LR, NB_EPOCH)





"""## Q2.3 MLP with Two Hidden Layer

Let $x \in R^{1\times N}$ be a single sample. $N$ is the number of the feature dimension and $C$ is the number of class in the final classification.

For the first layer, $W_1 \in R^{N\times \frac{N}{2}}$ and $b_1 \in R^{\frac{N}{2}}$

$$\bar{h}_1 = x\cdot W_1 + b_1$$

$$h_1 = ReLU(\bar{h}_1)$$

For the second layer, $W_2 \in R^{\frac{N}{2} \times \frac{N}{2}}$ and $b_2 \in R^{\frac{N}{2}}$

$$\bar{h}_2 = h_1\cdot W_2 + b_2$$

$$h_2 = h_1 + ReLU(\bar{h}_2)$$

*Sidenote: in deep learning, $g(x, f) = x + f(x)$ is usually referred to as skip connection.*

In the softmax layer, $W_{sm} \in R^{\frac{N}{2} \times C}$. The unnormalized score $s \in R^{C}$ is

$$s = h_2\cdot W_{sm}$$

With softmax function, we get the probability $p(c_i)$ of class $i$ where $i \in \{1,\cdots,C\}$

$$p(c_i) = \frac {\exp(s_i)} {\sum_i \exp(s_i)}$$

where $s_i$ is the $i^{th}$ element of $s$.

Assuming $y \in \{1,\cdots,C\}$, we want to minimize $-\log p(c_y)$, the cross-entropy loss between one hot true distribution `q(c_i) = 1 if i == y else 0` and $p(c_i)$.

We will take the average $-\log p(c_y)$ of a batch of x as the loss.

In `mlp_two_hidden_forward_and_backward`, `batch_x` and `batch_y` are matrixs, where each row are $x$ and $y$, respectively.
"""

mlp_two_hidden = Parameters()
mlp_two_hidden.set_param('w1', init_linear(NB_FEAT, NB_FEAT // 2))
mlp_two_hidden.set_param('b1', np.zeros(NB_FEAT // 2))
mlp_two_hidden.set_param('w2', init_linear(NB_FEAT // 2, NB_FEAT // 2))
mlp_two_hidden.set_param('b2', np.zeros(NB_FEAT // 2))
mlp_two_hidden.set_param('w_sm', init_linear(NB_FEAT // 2, 10))

"""### TODO"""

def relu(x):
    return np.maximum(0, x)

def mlp_two_hidden_forward_and_backward(batch_x, batch_y, param: Parameters, grad=True):
    '''
    compute the loss and the gradient of each parameter
    return loss (float) and p(c_i) (matrix with shape batch_size x C)
    '''
    param.zero_grad()
    w1 = param.get_param('w1')
    b1 = param.get_param('b1')
    w2 = param.get_param('w2')
    b2 = param.get_param('b2')
    w_sm = param.get_param('w_sm')

    #print(batch_x.shape)
    #print(batch_y.shape)
    #print(w1.shape)
    #print(b1.shape)
    #print(w2.shape)
    #print(b2.shape)
    #print(w_sm.shape)

    # compute loss and probs
    loss = 0
    probs = np.zeros((len(batch_x), 10))
    h1_bar = batch_x @ w1 + b1.reshape(1, -1)
    h1 = relu(h1_bar)
    h2_bar = h1 @ w2 + b2.reshape(1, -1)
    h2 = h1 + relu(h2_bar)
    #print(h1.shape)
    batch_s = h2 @ w_sm
    logprobs = batch_s - row_logsumexp(batch_s)
    probs = np.exp(logprobs)
    loss = -logprobs[np.arange(len(batch_y)), batch_y].mean()
    #raise NotImplementedError

    if not grad:
        return loss, probs

    # compute gradient
    w1_grad = np.zeros_like(w1)
    b1_grad = np.zeros_like(b1)
    w2_grad = np.zeros_like(w2)
    b2_grad = np.zeros_like(b2)
    w_sm_grad = np.zeros_like(w_sm)

    probs_copy = np.copy(probs)
    probs[np.arange(len(batch_y)),batch_y] -= 1
    probs /= len(batch_y)
    w_sm_grad = h2.T @ probs
    h2_grad = probs @ w_sm.T
    h2_bar_grad = np.copy(h2_grad)
    index_of_h2_bar_smaller_than_0 = np.where(h2_bar < 0)
    #print(index_of_h1_bar_smaller_than_0)
    h2_bar_grad[index_of_h2_bar_smaller_than_0] = 0
    w2_grad = h1.T @ h2_bar_grad
    b2_grad = h2_bar_grad.sum(axis = 0)

    h1_grad = h2_bar_grad @ w2.T + h2_grad
    index_of_h1_bar_smaller_than_0 = np.where(h1_bar < 0)
    #print(index_of_h1_bar_smaller_than_0)
    h1_grad[index_of_h1_bar_smaller_than_0] = 0
    
    w1_grad = batch_x.T @ h1_grad
    b1_grad = h1_grad.sum(axis = 0)

    probs = np.copy(probs_copy)

    param.accumlate_grad('w1', w1_grad)
    param.accumlate_grad('b1', b1_grad)
    param.accumlate_grad('w2', w2_grad)
    param.accumlate_grad('b2', b2_grad)
    param.accumlate_grad('w_sm', w_sm_grad)
    return loss, probs

"""### Loss & Gradient Check

If all assertions are passed, it means you are good to go.
"""

# use a prime number for batch size would make debugging easier
bs = 7
batch_x = train_x[:bs]
batch_y = train_y[:bs]
loss, _ = mlp_two_hidden_forward_and_backward(batch_x, batch_y, mlp_two_hidden)

w1 = mlp_two_hidden.get_param('w1')
b1 = mlp_two_hidden.get_param('b1')
w2 = mlp_two_hidden.get_param('w2')
b2 = mlp_two_hidden.get_param('b2')
w_sm = mlp_two_hidden.get_param('w_sm')
_w1 = torch.tensor(w1, dtype=torch.double, requires_grad=True)
_b1 = torch.tensor(b1, dtype=torch.double, requires_grad=True)
_w2 = torch.tensor(w2, dtype=torch.double, requires_grad=True)
_b2 = torch.tensor(b2, dtype=torch.double, requires_grad=True)
_w_sm = torch.tensor(w_sm, dtype=torch.double, requires_grad=True)
_batch_x = torch.tensor(batch_x, dtype=torch.double, requires_grad=False)
_batch_y = torch.tensor(batch_y, dtype=torch.long, requires_grad=False)

_h1 = F.linear(_batch_x, _w1.T) + _b1
_h1 = F.relu(_h1)

_h2 = F.linear(_h1, _w2.T) + _b2
_h2 = F.relu(_h2)
_h2 = _h2 + _h1

_logits = F.linear(_h2, _w_sm.T)
_logprobs = F.log_softmax(_logits, dim=-1)
_loss = -_logprobs[torch.arange(len(_batch_y)), _batch_y].mean()
_loss.backward()

assert np.isclose(loss, _loss.item())
assert np.allclose(_w_sm.grad.numpy(), mlp_two_hidden.grad['w_sm'])
assert np.allclose(_w2.grad.numpy(), mlp_two_hidden.grad['w2'])
assert np.allclose(_b2.grad.numpy(), mlp_two_hidden.grad['b2'])
assert np.allclose(_w1.grad.numpy(), mlp_two_hidden.grad['w1'])
assert np.allclose(_b1.grad.numpy(), mlp_two_hidden.grad['b1'])

"""### Train!"""

BS = 50
LR = 0.005
NB_EPOCH = 20

mlp_two_hidden = Parameters()
mlp_two_hidden.set_param('w1', init_linear(NB_FEAT, NB_FEAT // 2))
mlp_two_hidden.set_param('b1', np.zeros(NB_FEAT // 2))
mlp_two_hidden.set_param('w2', init_linear(NB_FEAT // 2, NB_FEAT // 2))
mlp_two_hidden.set_param('b2', np.zeros(NB_FEAT // 2))
mlp_two_hidden.set_param('w_sm', init_linear(NB_FEAT // 2, 10))

MLP_with_2_hidden_layers_dev_acc = main_training(mlp_two_hidden, mlp_two_hidden_forward_and_backward, train_x, train_y, dev_x, dev_y, BS, LR, NB_EPOCH)

BS = 100
LR = 0.01
NB_EPOCH = 20

mlp_two_hidden = Parameters()
mlp_two_hidden.set_param('w1', init_linear(NB_FEAT, NB_FEAT // 2))
mlp_two_hidden.set_param('b1', np.zeros(NB_FEAT // 2))
mlp_two_hidden.set_param('w2', init_linear(NB_FEAT // 2, NB_FEAT // 2))
mlp_two_hidden.set_param('b2', np.zeros(NB_FEAT // 2))
mlp_two_hidden.set_param('w_sm', init_linear(NB_FEAT // 2, 10))
My_train_dev_acc = main_training(mlp_two_hidden, mlp_two_hidden_forward_and_backward, train_x, train_y, dev_x, dev_y, BS, LR, NB_EPOCH)

import matplotlib.pyplot as plt
t = range(20)
plt.plot(t, one_linear_classifier_dev_loss, label = 'One_linear_classifier_dev_loss')
plt.plot(t, MLP_with_single_hidden_layer_dev_loss, label = 'MLP_with_single_hidden_layer_dev_loss')
plt.plot(t, MLP_with_2_hidden_layers_dev_acc, label = 'MLP_with_2_hidden_layers_dev_acc')
#plt.plot(t, My_train_dev_acc, label = 'My_train_dev_acc')
plt.legend()

import matplotlib.pyplot as plt
t = range(20)
plt.plot(t, one_linear_classifier_dev_loss, label = 'One_linear_classifier_dev_loss')
plt.plot(t, MLP_with_single_hidden_layer_dev_loss, label = 'MLP_with_single_hidden_layer_dev_loss')
plt.plot(t, MLP_with_2_hidden_layers_dev_acc, label = 'MLP_with_2_hidden_layers_dev_acc')
plt.plot(t, My_train_dev_acc, label = 'My_train_dev_acc')
plt.legend()

test_loss, test_probs = mlp_two_hidden_forward_and_backward(test_x, test_y, mlp_two_hidden, grad = False)
pred_y = test_probs.argmax(axis = 1)
test_acc = np.mean(pred_y == test_y) * 100
print(f'The test accuracy is {test_acc:.2f}%')