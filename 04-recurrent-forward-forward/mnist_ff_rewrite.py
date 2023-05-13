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
ITERATIONS = 10
THRESHOLD = .25
LEARNING_RATE = 0.0001


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

class Layer:
    def __init__():
        pass

    # TODO: figure out the best way to model the input and output layers
    def train_timestep():
        pass

class RecurrentFFNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes, damping_factor=0.7):
        super(RecurrentFFNet, self).__init__()

    def train():
        pass


