"""
Implementation for MNIST with simple SNN.

Uses STDP learning for first two layers, and gd for last layer.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
import torch.utils.data as data
import torchvision

from spikingjelly.activation_based import learning, layer, neuron, functional


def MNIST_loaders(batch_size):
    train_dataset = torchvision.datasets.MNIST(
        root="./data",
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=True
    )
    test_dataset = torchvision.datasets.MNIST(
        root="./data",
        train=False,
        transform=torchvision.transforms.ToTensor(),
        download=True
    )

    train_data_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True
    )
    test_data_loader = data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        pin_memory=True
    )

    return train_data_loader, test_data_loader, train_dataset, test_dataset


def f_weight(x):
    return torch.clamp(x, -1, 1.)


if __name__ == "__main__":
    # this ensures that the current MacOS version is at least 12.3+
    print(torch.backends.mps.is_available())
    # this ensures that the current current PyTorch installation was built with MPS activated.
    print(torch.backends.mps.is_built())
    # set the output to device: mps
    device = torch.device("mps")

    lr = 0.1
    batch_size = 64

    # multi step mode
    step_mode = "m"

    # should bias towards disincentivizing spikes coming too early
    tau_pre = 2.
    tau_post = 100.

    # simple network
    # linear layers are the weights
    net = nn.Sequential(
        layer.Linear(784, 28, bias=False),
        neuron.IFNode(),
        layer.Linear(28, 28, bias=False),
        neuron.IFNode(),
        layer.Linear(28, 28, bias=False),
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

    # get data
    train_data_loader, test_data_loader, train_dataset, test_dataset = MNIST_loaders(
        batch_size)

    # test forward pass
    examples = enumerate(train_data_loader)
    batch_idx, (example_data, example_targets) = next(examples)

    print(example_data.shape)

    x_seq = (example_data > 0.5).float()
    x_seq = x_seq.view(x_seq.size(0), -1)
    print(x_seq.shape)
    y = net(x_seq)
    print("y: ", y.shape)

    print(batch_idx, example_data.shape, example_targets.shape)

    print("done")
