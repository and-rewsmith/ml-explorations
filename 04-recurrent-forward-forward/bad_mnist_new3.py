from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def MNIST_loaders(train_batch_size=50000, test_batch_size=10000):
    transform = Compose([
        ToTensor(),
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


class RecurrentLayer(nn.Module):
    def __init__(self, in_features, out_features, damping_factor=0.3):
        super().__init__()
        self.layer = nn.Linear(in_features, out_features)
        self.norm = nn.LayerNorm(out_features)
        self.damping_factor = damping_factor

    def forward(self, input, prev_activation, recurrent_input=None):
        if recurrent_input is None:
            new_act = F.linear(input, self.layer.weight)
        else:
            new_act = F.linear(input, self.layer.weight) + F.linear(recurrent_input, self.layer.weight.t())

        new_act = (1 - self.damping_factor) * prev_activation + self.damping_factor * new_act
        new_act = self.norm(new_act)
        new_act = F.relu(new_act)
        return new_act

    def train_step(self, positive_input, negative_input, optimizer, threshold):
        optimizer.zero_grad()

        # Calculate positive goodness
        positive_goodness = torch.sum(torch.square(positive_input))

        # Calculate the goodness for negative data
        negative_goodness = torch.sum(torch.square(negative_input))

        # Initialize layer loss
        layer_loss = 0

        # Adjust the weights to increase the goodness for positive data
        if positive_goodness < threshold:
            layer_loss += (threshold - positive_goodness)

        # Adjust the weights to decrease the goodness for negative data
        if negative_goodness > threshold:
            layer_loss += (negative_goodness - threshold)

        layer_loss.backward(retain_graph=True)
        optimizer.step()

        return layer_loss.item()


class RecurrentFFNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes, damping_factor=0.3):
        super(RecurrentFFNet, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes

        # Define the linear layers
        self.layers = nn.ModuleList()
        prev_size = input_size
        for size in hidden_sizes:
            self.layers.append(RecurrentLayer(prev_size, size, damping_factor))
            prev_size = size
        self.layers.append(nn.Linear(prev_size, num_classes))

    def forward_timestep(self, input_image, prev_activations, one_hot_labels):
        new_activations = []
        output_layer_weights = self.layers[-1].weight.t()
        for i, layer in enumerate(self.layers[:-1]):
            if i == 0:  # first layer gets the input image
                new_act = layer(input_image, prev_activations[i], prev_activations[i + 1])
            elif i == len(self.layers) - 2:  # last hidden layer gets the previous layer's activation and the one-hot labels
                new_act = layer(prev_activations[i - 1], prev_activations[i], one_hot_labels)
            else:  # other layers get activations from the layers above and below
                new_act = layer(prev_activations[i - 1], prev_activations[i], prev_activations[i + 1])
            
            new_activations.append(new_act)

        # The output layer gets its input from the last hidden layer
        output = self.layers[-1](new_activations[-1])
        new_activations.append(output)

        return new_activations

    def train(model, train_loader, optimizer, device, num_iterations=8, threshold=0.5):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Initial activations are zero
            activations = [torch.zeros_like(layer.layer.bias) for layer in model.layers[:-1]] + [torch.zeros((images.shape[0], model.num_classes), device=device)]
            
            one_hot_labels = torch.zeros_like(activations[-1]).scatter_(1, labels.view(-1, 1), 1).to(device)

            for _ in range(num_iterations):
                activations = model.forward_timestep(images, activations, one_hot_labels)

                # Train each layer independently
                for i, layer in enumerate(model.layers[:-1]):
                    positive_activations = activations[i]
                    one_hot_labels[:, labels] = 1 - one_hot_labels[:, labels]
                    negative_activations = layer(images if i == 0 else activations[i - 1], activations[i], one_hot_labels if i == len(model.layers) - 2 else activations[i + 1])
                    layer_loss = layer.train_step(positive_activations, negative_activations, optimizer, threshold)
                    running_loss += layer_loss

        return running_loss / len(train_loader)

if __name__ == "__main__":
    torch.manual_seed(1234)
    train_loader, test_loader = MNIST_loaders()

    model = RecurrentFFNet(784, [500, 250, 100], 10).to(device)
    optimizer = Adam(model.parameters())
    num_epochs = 50

    for epoch in range(num_epochs):
        train(model, train_loader, optimizer, device)
