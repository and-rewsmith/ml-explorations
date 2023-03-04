from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torch

"""
Criticism of other FF impl to keep in mind:

The predict method computes the "goodness" score for each possible label by
iterating over all 10 labels and computing the score for each label separately.
This approach can be very slow and inefficient, especially for large datasets or
complex models. A more efficient approach would be to compute the scores for all
labels simultaneously using matrix multiplication.

The train method uses a fixed threshold of 2.0 for the "goodness" scores of the
positive and negative examples. This threshold may not be optimal for all
datasets or models, and may lead to suboptimal performance. A better approach
would be to use a dynamic threshold that adapts to the distribution of the
"goodness" scores during training.

The train method does not use any regularization techniques, such as weight
decay or dropout, to prevent overfitting. This may lead to overfitting of the
model to the training data, especially if the number of epochs is high.

 The train method does not use any validation or test sets to monitor the
 performance of the model during training. This may make it difficult to
 determine when the model has converged or whether it is overfitting.
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

if __name__ == "__main__":
    pass