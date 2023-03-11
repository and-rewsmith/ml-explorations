"""
Implementation from here:
https://spikingjelly.readthedocs.io/zh_CN/latest/activation_based_en/stdp.html
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
from spikingjelly.activation_based import learning, layer, neuron, functional

# TODO: figure out
T = 8
N = 2
C = 3
H = 32
W = 32
lr = 0.1
tau_pre = 2.
tau_post = 100.
step_mode = 'm'


def f_weight(x):
    return torch.clamp(x, -1, 1.)


# TODO: Figure out context on problem I see with this:
# 1. We are passing the whole matrix of timestamps into this network as multistep, but since we are using a combination of STDP and gradient descent, this requires the layer size to be arbitrarily high for non STDP layers which might otherwise be able to avoid it in certain scenarios.
# 2. Adding recurrent connections to IFNode will (I think) make the backprop needed for the gradient based learning harder, introducing need for BPTT.
net = nn.Sequential(
    layer.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
    neuron.IFNode(),
    layer.MaxPool2d(2, 2),
    layer.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
    neuron.IFNode(),
    layer.MaxPool2d(2, 2),
    layer.Flatten(),
    layer.Linear(16 * 8 * 8, 64, bias=False),
    neuron.IFNode(),
    layer.Linear(64, 10, bias=False),
    neuron.IFNode(),
)

functional.set_step_mode(net, step_mode)

instances_stdp = (layer.Conv2d, )

stdp_learners = []

for i in range(net.__len__()):
    if isinstance(net[i], instances_stdp):
        stdp_learners.append(
            learning.STDPLearner(step_mode=step_mode, synapse=net[i], sn=net[i+1], tau_pre=tau_pre, tau_post=tau_post,
                                 f_pre=f_weight, f_post=f_weight)
        )

params_stdp = []
for m in net.modules():
    if isinstance(m, instances_stdp):
        for p in m.parameters():
            params_stdp.append(p)
params_stdp_set = set(params_stdp)

params_gradient_descent = []
for p in net.parameters():
    if p not in params_stdp_set:
        params_gradient_descent.append(p)

optimizer_gd = Adam(params_gradient_descent, lr=lr)
optimizer_stdp = SGD(params_stdp, lr=lr, momentum=0.)

x_seq = (torch.rand([T, N, C, H, W]) > 0.5).float()
target = torch.randint(low=0, high=10, size=[N])

print("x_seq: ", x_seq.shape)
print("target: ", target)
print()
print("params_stdp: ")
for p in params_stdp:
    print(p.shape)
print()
print("params_gradient_descent: ")
for p in params_gradient_descent:
    print(p.shape)
print()

optimizer_gd.zero_grad()
optimizer_stdp.zero_grad()

y = net(x_seq).mean(0)
loss = F.cross_entropy(y, target)
loss.backward()
# zero gradients for non backprop trained layers
optimizer_stdp.zero_grad()

print("predicted: ", y)
print()
print("loss: ", loss)
print()

for i in range(stdp_learners.__len__()):
    stdp_learners[i].step(on_grad=True)

optimizer_gd.step()
optimizer_stdp.step()

print("params_stdp: ")
for p in params_stdp:
    print(p.shape)
print()
print("params_gradient_descent: ")
for p in params_gradient_descent:
    print(p.shape)
print()
