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
from tqdm import tqdm

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


# TODO: multiple epochs each with multiple batches
if __name__ == "__main__":
    # this ensures that the current MacOS version is at least 12.3+
    assert torch.backends.mps.is_available()
    # this ensures that the current current PyTorch installation was built with MPS activated.
    assert torch.backends.mps.is_built()
    # set the output to device: mps
    device = torch.device("mps")

    epochs = 2
    lr = 0.1
    batch_size = 64

    # if this changes we need to alter network input sizes in network and out
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
        layer.Linear(28, 10, bias=False),
        neuron.IFNode(),
    ).to(device)

    # train linear layer with stdp and ifnode with sgd
    functional.set_step_mode(net, step_mode)

    stdp_learners = []

    # last layer train with sgd
    instances_stdp = (layer.Conv2d, )
    for i in range(0, net.__len__() - 1):
        if isinstance(net[i], instances_stdp):
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
    for batch_idx, (example_data, example_targets) in tqdm(enumerate(train_data_loader)):
        example_data = example_data.to(device)
        example_targets = example_targets.to(device)
        # targets_onehot = torch.nn.functional.one_hot(example_targets)
        print("batch_idx: " + str(batch_idx))
        print("example_data: " + str(example_data.shape))
        print("example_targets: " + str(example_targets.shape))

        optimizer_stdp.zero_grad()
        optimizer_stdp.zero_grad()

        x_seq = (example_data > 0.5).float()
        # convert to sensible shape
        x_seq = x_seq.view(x_seq.size(0), -1)
        # convert to multistep
        T = 2
        x_seq = torch.unsqueeze(x_seq, 0).repeat(T, 1, 1)
        print("x_resize: ", x_seq.shape)

        print("x: " + str(x_seq.shape))
        y = functional.multi_step_forward(x_seq, net)
        y = torch.mean(y, dim=0)
        print("y: ", y.shape)
        _, predicted = torch.max(y, dim=1)
        print("predicted: ", predicted)

        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(y, example_targets)
        print("loss: " + str(loss))

        loss.backward()
        # zero gradients for non backprop trained layers
        optimizer_stdp.zero_grad()

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

        net.zero_grad()
        optimizer_stdp.zero_grad()
        optimizer_gd.zero_grad()

        functional.reset_net(net)

    print("done")
