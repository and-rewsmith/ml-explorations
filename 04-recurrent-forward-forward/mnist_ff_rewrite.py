import torch
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
import logging
import torchviz

# logging.basicConfig(filename="log.log", level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# QUESTIONS:
# 1 - Why does lowering threshold allow it to work?
# 2 - Layers are swapping between all high gooness predictions
#     Investigate layer activations?
#      Is there some bug?
#     Is model struggling to optimize parameter for multiple local layer trainings?
# TODO:
# 1 - Implement side connections as shown in Fig3
# 2 - Implement layer local training
#     - we can reference the pytorch ff example for this, also (https://github.com/cozheyuanzhangde/Forward-Forward/blob/main/model.py)
# 3 - Implement dedicated recurrent weights
# 4 - Experiment with more layers
# 5 - Experiment with weight initialization
# 6 - Experiment with activation initialization

ITERATIONS = 10
THRESHOLD = .25
LEARNING_RATE = 0.0001

"""
TIMELINE:
- Network class
- Network class has an init function
  + takes in input size, hidden sizes, num classes, and damping factor
  + create the layer class instances for all hidden layers
  + create a special InputLayer class instance and an OutputLayer class instance
- Network class has train function
  + takes in positive data, negative data, pos labels, neg labels
  + for layer in layers: layer.train()
- Network class has a test function
  + takes in data
  + for each possible label:
      for iteration in ITERATIONS:
          for layer in layers: layer.forward()
      calculate goodness and store
    return the label with the highest goodness
- Layer class
- Layer class has an init function
  + takes in prev_size, size, previous layer, next layer, damping factor
  + creates its own weight matrices (creating 2 weight matrices one for forward connection and one for backward)
  + initializes previous activations and current activations as all 0s
- Layer class has train function
  + takes in positive activations and negative activations
  + feeds in positive activations, gets output activations, calculates goodness, and does backwards + step
  + feeds in negative activations, gets output activations, calculates goodness, and does backwards + step
- Layer class has a forward function
  + takes in activations
  + uses its own weights, the previous layer's weights, and the previous timesteps activations to calculate the current activations
"""


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

class RecurrentFFNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes, damping_factor=0.7):
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

    def train():
        pass

    def test():
        pass

class Layer:
    def __init__(self):
        pass

    # TODO: figure out the best way to model the input and output layers
    def train():
        pass

    def forward():
        pass