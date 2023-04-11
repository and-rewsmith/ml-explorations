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
from torchvision.transforms import ToTensor, Normalize, Compose, Lambda

from tqdm import tqdm

from spikingjelly.activation_based import learning, layer, neuron, functional, encoding, surrogate


def flatten(x):
    return torch.flatten(x)


def MNIST_loaders(batch_size):
    # TODO: integrate transform
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


# TODO: figure out why conv2d layers get error "Runtime canonicalization must
#       simplify reduction axes to minor 4 dimensions." on mps
if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)

    # this ensures that the current MacOS version is at least 12.3+ and pytorch
    # has mps
    assert torch.backends.mps.is_available()
    assert torch.backends.mps.is_built()
    # set the output to device
    device = torch.device("mps")

    TIMESTEPS = 1000
    lr = 0.01
    batch_size = 64
    num_epochs = 10

    # if this changes we need to alter network input sizes in network and out
    step_mode = "m"

    # should bias towards disincentivizing spikes coming too early
    tau_pre = 2.
    tau_post = 100.

    # TODO: convert Conv2d layers to linear to simplify
    net = nn.Sequential(
        layer.Linear(784, 64, bias=False),
        nn.ReLU(),
        # neuron.IFNode(),
        layer.Linear(64, 64, bias=False),
        nn.ReLU(),
        # neuron.IFNode(),
        layer.Linear(64, 10, bias=False),
        # neuron.IFNode(),
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

    examples = enumerate(train_data_loader)
    for epoch in range(0, num_epochs):
        print("starting training for epoch 1")
        for batch_idx, (example_data, example_targets) in tqdm(enumerate(train_data_loader)):
            print()

            net.zero_grad()
            functional.reset_net(net)
            optimizer_gd.zero_grad()

            print("training for batch: ", batch_idx)
            x = example_data.to(device)
            example_targets = example_targets.to(device)
            # targets_onehot = torch.nn.functional.one_hot(
            #     example_targets, num_classes=10).float()

            # convert to sensible shape:
            T, D = x.shape
            x_unsqueezed = x.unsqueeze(1)
            x_repeat = x_unsqueezed.repeat(1, TIMESTEPS, 1)
            x = x_repeat.transpose(0, 1)

            # size: (TIMESTEPS, BATCH_SIZE, H*W)
            print("x: ", x.shape)

            y = functional.multi_step_forward(x, net)
            print("y: ", y.shape)
            y = torch.mean(y, dim=0)
            print("y: ", y.shape)
            # y = torch.argmax(y, dim=1)
            print("y: ", y.shape)

            # print("target probabilities: ", y.shape)
            print("prediction: ", str(torch.argmax(y, dim=1)[0]))
            print("actual: ", str(example_targets[0]))

            loss_fn = nn.CrossEntropyLoss()
            print("y: ", y.shape)
            print("targets: ", example_targets.shape)
            loss = loss_fn(y, example_targets)

            print("loss: " + str(loss))

            loss.backward()
            optimizer_gd.step()

            # print params to make sure they get updated between batches
            print("params_gradient_descent: ",
                  params_gradient_descent[0][0][0:5])

            for idx, p in enumerate(net.parameters()):
                print(f"Gradient for layer {idx}: {p.grad}")

        print("finished training for epoch")
