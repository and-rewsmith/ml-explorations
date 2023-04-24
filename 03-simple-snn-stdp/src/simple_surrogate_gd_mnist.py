"""
Implementation for MNIST with simple ANN and surrogage gradients for LIF nodes.

Serves as a foundation where we can extend this with unsupervised STDP training.
"""

# TODO: 4-23
# figure out why this network is not learning at all(loss not decreasing)
# there are some commented out lines that print some context

import torchvision

import torch
import torch.nn as nn
from torch.optim import Adam
import torch.utils.data as data
from torchvision.transforms import ToTensor, Normalize, Compose, Lambda
import torch.nn.functional as F

from tqdm import tqdm

from spikingjelly.activation_based import layer, functional, neuron, surrogate, encoding


def flatten(x):
    return torch.flatten(x)


def MNIST_loaders(batch_size):
    transform = Compose([
        ToTensor(),
        # This is well known for MNIST, so just use it.
        Normalize((0.1307,), (0.3081,)),
        Lambda(flatten)])

    train_dataset = torchvision.datasets.MNIST(
        root="./data",
        train=True,
        transform=transform,
        download=True,
    )
    test_dataset = torchvision.datasets.MNIST(
        root="./data",
        train=False,
        transform=transform,
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


"""
TODO:
- implement surrogate gradient
- encoder

"""
if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)

    # this ensures that the current MacOS version is at least 12.3+ and pytorch
    # has mps
    assert torch.backends.mps.is_available()
    assert torch.backends.mps.is_built()
    device = torch.device("cpu")

    TIMESTEPS = 100
    lr = 0.01
    batch_size = 64
    num_epochs = 1

    # if this changes we need to alter network input sizes in network and out
    step_mode = "m"

    tau = 2.

    net = nn.Sequential(
        layer.Linear(784, 64, bias=False),
        nn.BatchNorm1d(64),
        neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan()),
        layer.Linear(64, 64, bias=False),
        nn.BatchNorm1d(64),
        neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan()),
        layer.Linear(64, 10, bias=False),
        nn.BatchNorm1d(10),
        neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan()),
    ).to(device)

    functional.set_step_mode(net, step_mode)
    net.train()

    params_gradient_descent = []
    for p in net.parameters():
        print("gd param: " + str(p.shape))
        params_gradient_descent.append(p)

    optimizer_gd = Adam(params_gradient_descent, lr=lr)

    train_data_loader, test_data_loader, train_dataset, test_dataset = MNIST_loaders(
        batch_size)

    encoder = encoding.PoissonEncoder()

    examples = enumerate(train_data_loader)
    for epoch in range(0, num_epochs):
        print("starting training for epoch " + str(epoch))
        for batch_idx, (example_data, example_targets) in tqdm(enumerate(train_data_loader)):
            print()

            net.zero_grad()
            functional.reset_net(net)
            optimizer_gd.zero_grad()

            print("training for batch: ", batch_idx)
            x = example_data.to(device)
            example_targets = example_targets.to(device)
            targets_onehot = torch.nn.functional.one_hot(
                example_targets, num_classes=10).float()

            x = encoder(x)

            # convert to sensible shape:
            T, D = x.shape
            x_unsqueezed = x.unsqueeze(1)
            x_repeat = x_unsqueezed.repeat(1, TIMESTEPS, 1)
            x = x_repeat.transpose(0, 1)

            y = functional.multi_step_forward(x, net)
            y = torch.mean(y, dim=0)

            # print("y: " + str(y))

            print("first prediction: ", str(torch.argmax(y, dim=1)[0]))
            print("first actual: ", str(example_targets[0]))

            # loss_fn = nn.CrossEntropyLoss()
            # loss = loss_fn(y, example_targets)

            loss = F.mse_loss(y, targets_onehot).float()

            print("loss: " + str(loss))

            loss.backward()
            optimizer_gd.step()

            # print params to make sure they get updated between batches
            # print("params_gradient_descent: ",
            #       params_gradient_descent[0][0][0:5])

            # for idx, p in enumerate(net.parameters()):
            #     for row in p.grad:
            #         print(f"Gradient for layer {idx}: {row}")

        print("finished training for epoch " + str(epoch))
