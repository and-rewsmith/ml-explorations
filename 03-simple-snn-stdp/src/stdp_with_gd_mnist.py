"""
Implementation for MNIST with simple SNN.

Uses hybrid approach where first layers trained through STDP and last layers
trained with SGD.
"""

import torchvision

import torch
import torch.nn as nn
from torch.optim import SGD, Adam
import torch.utils.data as data

from tqdm import tqdm

from spikingjelly.activation_based import learning, layer, neuron, functional, encoding, surrogate


def MNIST_loaders(batch_size):
    # TODO: integrate transform
    # transform = Compose([
    #     ToTensor(),
    #     # This is well known for MNIST, so just use it.
    #     Normalize((0.1307,), (0.3081,)),
    #     Lambda(lambda x: torch.flatten(x))])

    train_dataset = torchvision.datasets.MNIST(
        root="./data",
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=True,
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


# TODO: figure out why conv2d layers get error "Runtime canonicalization must
#       simplify reduction axes to minor 4 dimensions." on mps
if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)

    # this ensures that the current MacOS version is at least 12.3+ and pytorch
    # has mps
    assert torch.backends.mps.is_available()
    assert torch.backends.mps.is_built()
    # set the output to device
    device = torch.device("cpu")

    TIMESTEPS = 100
    lr = 0.1
    batch_size = 64
    num_epochs = 1

    # if this changes we need to alter network input sizes in network and out
    step_mode = "m"

    # should bias towards disincentivizing spikes coming too early
    tau_pre = 2.
    tau_post = 100.

    # TODO: convert Conv2d layers to linear to simplify
    net = nn.Sequential(
        layer.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False),
        neuron.IFNode(),
        layer.MaxPool2d(2, 2),
        layer.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False),
        neuron.IFNode(),
        layer.MaxPool2d(2, 2),
        layer.Flatten(),
        layer.Linear(784, 64, bias=False),
        neuron.IFNode(),
        layer.Linear(64, 10, bias=False),
        neuron.IFNode(),
    ).to(device)

    functional.set_step_mode(net, step_mode)
    net.train()

    # all but last 2 layers train with stdp
    stdp_learners = []
    instances_stdp = (layer.Conv2d, )
    for i in range(0, net.__len__()):
        if isinstance(net[i], instances_stdp):
            stdp_learners.append(
                learning.STDPLearner(step_mode=step_mode, synapse=net[i], sn=net[i+1], tau_pre=tau_pre, tau_post=tau_post,
                                     f_pre=f_weight, f_post=f_weight))

    # last 2 layers train with backprop
    params_stdp = []
    for m in net.modules():
        if isinstance(m, instances_stdp):
            for p in list(m.parameters()):
                print("stdp param: " + str(p.shape))
                params_stdp.append(p)
    params_stdp_set = set(params_stdp)

    params_gradient_descent = []
    for p in net.parameters():
        if p not in params_stdp_set:
            print("gd param: " + str(p.shape))
            params_gradient_descent.append(p)

    optimizer_gd = Adam(params_gradient_descent, lr=lr)
    optimizer_stdp = SGD(params_stdp, lr=lr, momentum=0.)

    train_data_loader, test_data_loader, train_dataset, test_dataset = MNIST_loaders(
        batch_size)

    encoder = encoding.PoissonEncoder()
    examples = enumerate(train_data_loader)
    for epoch in range(0, num_epochs):
        print("starting training for epoch 1")
        for batch_idx, (example_data, example_targets) in tqdm(enumerate(train_data_loader)):
            print()

            functional.reset_net(net)
            optimizer_stdp.zero_grad()
            optimizer_gd.zero_grad()

            print("training for batch: ", batch_idx)
            x_seq = example_data.to(device)
            example_targets = example_targets.to(device)
            targets_onehot = torch.nn.functional.one_hot(
                example_targets).float()

            # convert to sensible shape:
            # [Time, batches, channel size, height, width]
            # = [Time, T, C, H, W]
            x_seq = x_seq.view(x_seq.size(0), -1)
            x_seq = torch.unsqueeze(x_seq, 0).repeat(TIMESTEPS, 1, 1)
            x_seq = x_seq.view(TIMESTEPS, 64, 1, 28, 28)
            x_seq = encoder(x_seq)

            # at this point we have tensor of shape: [Time, T, C, H, W]
            # we need to reshape to add channel size: [Time, T, N, C, H, W]
            print("x input: " + str(x_seq.shape))
            Z, T, C, H, W = x_seq.shape
            N = 1
            x_seq = x_seq.unsqueeze(2)
            x_seq.view(Z, T, N, C, H, W)
            print("x reshaped: " + str(x_seq.shape))

            y = functional.multi_step_forward(x_seq, net)
            y = torch.mean(y, dim=0)
            y = y.squeeze(1)
            y = torch.softmax(y, dim=1)

            print("target probabilities: ", y.shape)

            loss_fn = nn.BCELoss()
            loss = loss_fn(y, targets_onehot)

            print("loss: " + str(loss))

            loss.backward()

            # zero gradients for non sgd trained layers
            optimizer_stdp.zero_grad()

            for i in range(stdp_learners.__len__()):
                stdp_learners[i].step(on_grad=True)

            optimizer_gd.step()
            optimizer_stdp.step()

            # print params to make sure they get updated between batches
            print("params_gradient_descent: ",
                  params_gradient_descent[0][0][0:5])

        print("finished training for epoch 1")
