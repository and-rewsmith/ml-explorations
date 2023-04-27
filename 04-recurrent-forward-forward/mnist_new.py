import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader

"""
TODO:
Tune threshold.
Tune damping factor.
Tune number of iterations.
Tune batch size.

Testing
"""

# this ensures that the current MacOS version is at least 12.3+
print(torch.backends.mps.is_available())
# this ensures that the current current PyTorch installation was built with MPS activated.
print(torch.backends.mps.is_built())
# set the output to device: mps
device = torch.device("mps")

def MNIST_loaders(train_batch_size=50000, test_batch_size=10000):

    transform = Compose([
        ToTensor(),
        # This is well known for MNIST, so just use it.
        Normalize((0.1307,), (0.3081,)),
        Lambda(lambda x: torch.flatten(x))])

    train_loader = DataLoader(
        MNIST('./data/', train=True,
              download=True,
              transform=transform),
        batch_size=train_batch_size, shuffle=True)

    test_loader = DataLoader(
        MNIST('./data/', train=False,
              download=True,
              transform=transform),
        batch_size=test_batch_size, shuffle=False)

    return train_loader, test_loader


class RecurrentFFNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes, damping_factor=0.3):
        super(RecurrentFFNet, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes
        self.damping_factor = damping_factor

        # Define the linear layers
        self.layers = nn.ModuleList()
        prev_size = input_size
        for size in hidden_sizes:
            self.layers.append(nn.Linear(prev_size, size))
            prev_size = size
        self.layers.append(nn.Linear(prev_size, num_classes))

        # Define layer normalization for hidden layers
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(size) for size in hidden_sizes])

    def forward_once(self, x):
        activations = []
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.layer_norms[i](x)
            x = F.relu(x)
            activations.append(x)
        return activations

    def forward_recurrent(self, prev_activations):
        new_activations = []
        for i, (prev_act, layer, norm) in enumerate(zip(prev_activations, self.layers[:-1], self.layer_norms)):
            if i == 0:
                new_act = layer(prev_activations[i + 1])
            elif i == len(prev_activations) - 1:
                new_act = layer(prev_activations[i - 1])
            else:
                new_act = layer(prev_activations[i - 1] + prev_activations[i + 1])
            new_act = (1 - self.damping_factor) * prev_act + self.damping_factor * new_act
            new_act = norm(new_act)
            new_act = F.relu(new_act)
            new_activations.append(new_act)
        return new_activations

    def forward_with_iterations(self, input_image, label, num_iterations=8):
        # Initialize the hidden layers' activities with forward_once
        activations = self.forward_once(input_image)

        # Set the initial output layer activation using the label (one-hot encoded)
        output_layer_activation = F.one_hot(label, num_classes=self.num_classes).float()
        
        # Perform multiple iterations of the recurrent forward pass
        for _ in range(num_iterations):
            activations = [output_layer_activation] + activations[:-1]
            activations = self.forward_recurrent(activations)

        # Return the activations of all layers or just the hidden layers
        return activations

def train(model, train_loader, optimizer, device, num_iterations=8, threshold=0.5):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Calculate the goodness for positive data
        one_hot_labels = torch.zeros_like(labels).scatter_(1, labels.view(-1, 1), 1).to(device)
        positive_activations = model.forward_with_iterations(images, one_hot_labels, num_iterations)
        positive_goodness = [torch.sum(torch.square(layer_activations)) for layer_activations in positive_activations]

        # Calculate the goodness for negative data
        one_hot_labels[:, labels] = 1 - one_hot_labels[:, labels]
        negative_activations = model.forward_with_iterations(images, one_hot_labels, num_iterations)
        negative_goodness = [torch.sum(torch.square(layer_activations)) for layer_activations in negative_activations]

        # Train each layer independently
        for i, layer in enumerate(model.layers[:-1]):
            optimizer.zero_grad()
            layer_loss = 0

            # Adjust the weights to increase the goodness for positive data
            if positive_goodness[i] < threshold:
                layer_loss += (threshold - positive_goodness[i])

            # Adjust the weights to decrease the goodness for negative data
            if negative_goodness[i] > threshold:
                layer_loss += (negative_goodness[i] - threshold)

            layer_loss.backward()
            optimizer.step()

            running_loss += layer_loss.item()

    return running_loss / len(train_loader)

if __name__ == "__main__":
    torch.manual_seed(1234)
    train_loader, test_loader = MNIST_loaders()

    model = RecurrentFFNet(784, [500, 250, 100], 10).to(device)
    optimizer = Adam(model.parameters())
    criterion = nn.MSELoss()
    num_epochs = 50

    # Train the model
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer, device)
        print(f"Epoch {epoch + 1}, Loss: {train_loss:.4f}")
