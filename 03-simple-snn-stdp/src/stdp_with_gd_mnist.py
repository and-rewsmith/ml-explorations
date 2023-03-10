"""
Implementation for MNIST with simple SNN.

Uses STDP learning for first two layers, and gd for last layer.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
from spikingjelly.activation_based import learning, layer, neuron, functional

def f_weight(x):
    return torch.clamp(x, -1, 1.)

if __name__ == "__main__":
    # multi step mode
    step_mode = "m"

    # TODO: play with learning rate
    lr = 0.1

    # should bias towards disincentivizing spikes coming too early
    tau_pre = 2.
    tau_post = 100.

    # simple network
    # linear layers are the weights
    net = nn.Sequential(
        layer.Linear(784, 10, bias=False),
        neuron.IFNode(),
        layer.Linear(784, 10, bias=False),
        neuron.IFNode(),
        layer.Linear(784, 10, bias=False),
        neuron.IFNode(),
    )

    # train linear layer with stdp and ifnode with sgd
    functional.set_step_mode(net, step_mode)

    stdp_learners = []

    # last layer train with sgd
    for i in range(0, net.__len__() - 1):
        stdp_learners.append(
            learning.STDPLearner(step_mode=step_mode, synapse=net[i], sn=net[i+1], tau_pre=tau_pre, tau_post=tau_post,
                                f_pre=f_weight, f_post=f_weight))
            

    # last layer train with sgd
    params_stdp = []
    for m in net.modules():
        for p in list(m.parameters())[:-1]:
            print("stdp param: " + str(p.shape))
            params_stdp.append(p)
    params_stdp_set = set(params_stdp)

    params_gradient_descent = []
    for p in net.parameters():
        print("net param: " + str(p.shape))
        if p not in params_stdp_set:
            params_gradient_descent.append(p)

    optimizer_gd = Adam(params_gradient_descent, lr=lr)
    optimizer_stdp = SGD(params_stdp, lr=lr, momentum=0.)
    
    print("done")