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

    TIMESTEPS = 2
    lr = 0.1
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
        # neuron.IFNode(),
        layer.Linear(64, 64, bias=False),
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

            functional.reset_net(net)
            optimizer_gd.zero_grad()

            print("training for batch: ", batch_idx)
            x = example_data.to(device)
            example_targets = example_targets.to(device)
            targets_onehot = torch.nn.functional.one_hot(
                example_targets, num_classes=10).float()

            # TODO: This isnt' working due to something with x generation
            # convert to sensible shape:
            T, C, H, W = x.shape
            # Flatten the tensor along dimensions H and W
            x_flat = x.view(T, C, -1)
            # Duplicate the flattened tensor along the new dimension Z
            x_dup = x_flat.repeat_interleave(TIMESTEPS, dim=0)
            # Reshape the duplicated tensor to the desired shape [TIMESTEPS, T, H*W]
            x = x_dup.view(TIMESTEPS, T, -1)

            # size: (TIMESTEPS, BATCH_SIZE, H*W)
            print("x: ", x.shape)

            y = functional.multi_step_forward(x, net)
            print("y: ", y.shape)
            y = torch.mean(y, dim=0)
            print("y: ", y.shape)
            y = torch.softmax(y, dim=1)
            print("y: ", y.shape)

            # print("target probabilities: ", y.shape)
            print("prediction: ", str(torch.argmax(y[0], dim=0)))
            print("actual: ", str(torch.argmax(targets_onehot[0], dim=0)))

            loss_fn = nn.CrossEntropyLoss()
            print("y: ", y.shape)
            print("targets: ", targets_onehot.shape)
            loss = loss_fn(y, targets_onehot)

            print("loss: " + str(loss))

            loss.backward()
            optimizer_gd.step()

            # print params to make sure they get updated between batches
            print("params_gradient_descent: ",
                  params_gradient_descent[0][0][0:5])

        print("finished training for epoch 1")
