from enum import Enum
import logging

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
import torchviz
import wandb

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# QUESTIONS:
# 1 - First layer activations get very sparse?

# TODO:
# 1 - Implement side connections as shown in Fig3
# 2 - Experiment with more layers
# 3 - Experiment with weight initialization
# 4 - Stop using nn.Linear if all we need are simple weights
# 5 - Try different optimizers
# 6 - Model.eval()? We are not using it now.
# 7 - try layer norm layers

EPOCHS = 1000000
ITERATIONS = 10
THRESHOLD = 1
DAMPING_FACTOR = 0.7
EPSILON = 1e-8
LEARNING_RATE = 0.00005
LAYERS = [500, 250, 200]

INPUT_SIZE = 784
NUM_CLASSES = 10


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


class InputData:
    def __init__(self, pos_input, neg_input):
        self.pos_input = pos_input
        self.neg_input = neg_input

    def __iter__(self):
        yield self.pos_input
        yield self.neg_input


class LabelData:
    def __init__(self, pos_labels, neg_labels):
        self.pos_labels = pos_labels
        self.neg_labels = neg_labels

    def __iter__(self):
        yield self.pos_labels
        yield self.neg_labels


class TestData:
    def __init__(self, input, one_hot_labels, labels):
        self.input = input
        self.one_hot_labels = one_hot_labels
        self.labels = labels

    def __iter__(self):
        yield self.input
        yield self.one_hot_labels
        yield self.labels


class Activations:
    def __init__(self, current, previous):
        self.current = current
        self.previous = previous

    def __iter__(self):
        yield self.current
        yield self.previous

    def advance(self):
        self.previous = self.current


class OutputLayer(nn.Module):
    def __init__(self, prev_size, label_size) -> None:
        super(OutputLayer, self).__init__()

        self.backward_linear = nn.Linear(
            label_size, prev_size)


class ForwardMode(Enum):
    PositiveData = 1
    NegativeData = 2
    PredictData = 3


# TODO: optimize this to not have lists
def activations_to_goodness(activations):
    """
    Computes the 'goodness' of activations for each layer in a neural network by
    taking the mean of the squared values.

    'Goodness' in this context refers to the average squared activation value.
    This function is designed to work with PyTorch tensors, which represent
    layers in a neural network.

    Args:
        activations (list of torch.Tensor): A list of tensors representing
        activations from each layer of a neural network. Each tensor in the list
        corresponds to one layer's activations, and has shape (batch_size,
        num_activations), where batch_size is the number of samples processed
        together, and num_activations is the number of neurons in the layer.

    Returns:
        list of torch.Tensor: A list of tensors, each tensor corresponding to
        the 'goodness' (mean of the squared activations) of each layer in the
        input. Each tensor in the output list has shape (batch_size,), since the
        mean is taken over the activation values for each sample in the batch.
    """
    goodness = []
    for act in activations:
        goodness_for_layer = torch.mean(
            torch.square(act), dim=1)
        goodness.append(goodness_for_layer)

    return goodness


def layer_activations_to_goodness(layer_activations):
    """
    Computes the 'goodness' of activations for a given layer in a neural network
    by taking the mean of the squared values.

    'Goodness' in this context refers to the average squared activation value.
    This function is designed to work with PyTorch tensors, which represent the
    layer's activations.

    Args:
        layer_activations (torch.Tensor): A tensor representing activations from
        one layer of a neural network. The tensor has shape (batch_size,
        num_activations), where batch_size is the number of samples processed
        together, and num_activations is the number of neurons in the layer.

    Returns:
        torch.Tensor: A tensor corresponding to the 'goodness' (mean of the
        squared activations) of the given layer. The output tensor has shape
        (batch_size,), since the mean is taken over the activation values for
        each sample in the batch.
    """
    goodness_for_layer = torch.mean(
        torch.square(layer_activations), dim=1)

    return goodness_for_layer


class RecurrentFFNet(nn.Module):
    """
    Implements a Recurrent Forward-Forward Network (RecurrentFFNet) based on
    PyTorch's nn.Module.

    This class represents a multi-layer network composed of an input layer, one
    or more hidden layers, and an output layer. Unlike traditional feed-forward
    networks, the hidden layers in this network are recurrent, i.e., they are
    connected back to themselves across timesteps. 

    The learning procedure used here is a variant of the "Forward-Forward"
    algorithm, which is a greedy multi-layer learning method inspired by
    Boltzmann machines and Noise Contrastive Estimation. Instead of a
    traditional forward and backward pass, this algorithm employs two forward
    passes operating on different data and with contrasting objectives.

    During training, a "positive" pass operates on real input data and adjusts
    the weights to increase the 'goodness' in each hidden layer. The 'goodness'
    is calculated as the sum of squared activation values. On the other hand, a
    "negative" pass operates on "negative data" and adjusts the weights to
    decrease the 'goodness' in each hidden layer.

    The hidden layers and output layer are instances of the HiddenLayer and
    OutputLayer classes, respectively. The hidden layers are connected to each
    other and the output layer, forming a fully connected recurrent
    architecture.
    """
    # Ties all the layers together.
    # TODO: look to see if I should be overriding reset_net()
    def __init__(self, train_batch_size, test_batch_size, input_size, hidden_sizes, num_classes, damping_factor=DAMPING_FACTOR):
        logging.info("initializing network")
        super(RecurrentFFNet, self).__init__()

        self.layers = nn.ModuleList()
        prev_size = input_size
        for size in hidden_sizes:
            hidden_layer = HiddenLayer(
                train_batch_size, test_batch_size, prev_size, size, damping_factor)
            self.layers.append(hidden_layer)
            prev_size = size

        self.output_layer = OutputLayer(hidden_sizes[-1], num_classes)

        # attach layers to each other
        for i, hidden_layer in enumerate(self.layers):
            if i != 0:
                hidden_layer.set_previous_layer(self.layers[i - 1])

        for i, hidden_layer in enumerate(self.layers):
            if i != len(self.layers) - 1:
                hidden_layer.set_next_layer(self.layers[i + 1])
            else:
                hidden_layer.set_next_layer(self.output_layer)

        self.optimizer = Adam(self.parameters(), lr=LEARNING_RATE)

        logging.info("finished initializing network")

    def reset_activations(self, isTraining):
        for layer in self.layers:
            layer.reset_activations(isTraining)

    def train(self, input_data, label_data, test_data):
        """
        Trains the RecurrentFFNet model using the provided input data and label
        data.

        Args:
            input_data (object): An object containing positive and negative
            input data tensors. 

            label_data (object): An object containing positive and negative
            label data tensors.

            test_data (tuple): A tuple containing the test data, one-hot labels,
            and the actual labels.

        Procedure:
            For each epoch, the method resets the network's activations and
            performs a preinitialization step, forwarding both positive and
            negative data through the network. It then runs a specified number
            of iterations, where it trains the network using the input and label
            data. 

            After each epoch, the method calculates the 'goodness' metric for
            each layer in the network (for both positive and negative data),
            averages the layer losses, and then calculates the prediction
            accuracy on the test data. 

            Finally, the method logs these metrics (accuracy, average loss, and
            layer-wise 'goodness' scores) for monitoring the training process.
            The metrics logged depend on the number of layers in the network.

        Note:
            'Goodness' here refers to a metric that indicates how well the
            model's current activations represent a given class. It's calculated
            by a function `layer_activations_to_goodness` (not shown in this
            code), which transforms a layer's activations into a 'goodness'
            score. This function operates on the RecurrentFFNet model level and
            is called during the training process.
        """
        for epoch in range(0, EPOCHS):
            logging.info("Epoch: " + str(epoch))
            self.reset_activations(True)

            for preinit_step in range(0, len(self.layers)):
                logging.info("Preinitialization step: " + str(preinit_step))
                self.__advance_layers_forward(ForwardMode.PositiveData,
                                              input_data.pos_input, label_data.pos_labels, False)
                self.__advance_layers_forward(ForwardMode.NegativeData,
                                              input_data.neg_input, label_data.neg_labels, False)

            for iteration in range(0, ITERATIONS):
                logging.info("Iteration: " + str(iteration))
                total_loss = self.__advance_layers_train(
                    input_data, label_data, True)
                average_layer_loss = (total_loss / len(self.layers)).item()
                logging.info("Average layer loss: " + str(average_layer_loss))


            pos_goodness_per_layer = [
                layer_activations_to_goodness(layer.pos_activations.current).mean() for layer in self.layers]
            neg_goodness_per_layer = [
                layer_activations_to_goodness(layer.neg_activations.current).mean() for layer in self.layers]

            # Supports wandb tracking of max 3 layer goodnesses    
            try:            
                first_layer_pos_goodness = pos_goodness_per_layer[0]
                first_layer_neg_goodness = neg_goodness_per_layer[0]
                second_layer_pos_goodness = pos_goodness_per_layer[1]
                second_layer_neg_goodness = neg_goodness_per_layer[1]
                third_layer_pos_goodness = pos_goodness_per_layer[2]
                third_layer_neg_goodness = neg_goodness_per_layer[2]
            except:
                # No-op as there may not be 3 layers
                pass

            accuracy = self.predict(test_data)

            if len(self.layers) == 3:
                wandb.log({"acc": accuracy, "loss": average_layer_loss, "first_layer_pos_goodness": first_layer_pos_goodness, "second_layer_pos_goodness": second_layer_pos_goodness, "third_layer_pos_goodness": third_layer_pos_goodness, "first_layer_neg_goodness": first_layer_neg_goodness, "second_layer_neg_goodness": second_layer_neg_goodness, "third_layer_neg_goodness": third_layer_neg_goodness})
            elif len(self.layers) == 2:
                wandb.log({"acc": accuracy, "loss": average_layer_loss, "first_layer_pos_goodness": first_layer_pos_goodness, "second_layer_pos_goodness": second_layer_pos_goodness, "first_layer_neg_goodness": first_layer_neg_goodness, "second_layer_neg_goodness": second_layer_neg_goodness})
            elif len(self.layers) == 1:
                wandb.log({"acc": accuracy, "loss": average_layer_loss, "first_layer_pos_goodness": first_layer_pos_goodness, "first_layer_neg_goodness": first_layer_neg_goodness})

    def predict(self, test_data):
        """
        This function predicts the class labels for the provided test data using
        the trained RecurrentFFNet model. 

        Args:
            test_data (object): A tuple containing the test data, one-hot labels,
            and the actual labels. The data is assumed to be PyTorch tensors.

        Returns:
            float: The prediction accuracy as a percentage.

        Procedure:
            The function first moves the test data and labels to the appropriate
            device. It then calculates the 'goodness' metric for each possible
            class label, using a two-step process:

                1. Resetting the network's activations and forwarding the data
                   through the network with the current label.
                2. For each iteration within a specified threshold, forwarding
                   the data again, but this time retaining the
                activations, which are used to calculate the 'goodness' for each
                layer.

            The 'goodness' values across iterations and layers are then averaged
            to produce a single 'goodness' score for each class label. The class
            with the highest 'goodness' score is chosen as the prediction for
            each test sample.

            Finally, the function calculates the overall accuracy of the model's
            predictions by comparing them to the actual labels and returns this
            accuracy.
        """
        with torch.no_grad():
            data, one_hot_labels, labels = test_data
            data = data.to(device)
            one_hot_labels = one_hot_labels.to(device)
            labels = labels.to(device)

            all_labels_goodness = []

            # evaluate goodness for each possible label
            for label in range(NUM_CLASSES):
                self.reset_activations(False)

                one_hot_labels = torch.zeros(
                    data.shape[0], NUM_CLASSES, device=device)
                one_hot_labels[:, label] = 1.0

                for _preinit in range(0, len(self.layers)):
                    self.__advance_layers_forward(ForwardMode.PredictData,
                                                  data, one_hot_labels, False)

                lower_iteration_threshold = ITERATIONS // 2 - 1
                upper_iteration_threshold = ITERATIONS // 2 + 1
                goodnesses = []
                for _iteration in range(0, ITERATIONS):
                    self.__advance_layers_forward(ForwardMode.PredictData,
                                                  data, one_hot_labels, True)
                    
                    if _iteration >= lower_iteration_threshold and  _iteration <= upper_iteration_threshold:
                        layer_goodnesses = []
                        for layer in self.layers:
                            layer_goodnesses.append(layer_activations_to_goodness(layer.predict_activations.current))

                        goodnesses.append(torch.stack(layer_goodnesses, dim=1))

                # tensor of shape (batch_size, iterations, num_layers) 
                goodnesses = torch.stack(goodnesses, dim=1)
                print(goodnesses.shape)
                # average over iterations
                goodnesses = goodnesses.mean(dim=1)
                print(goodnesses.shape)
                # average over layers
                goodness = goodnesses.mean(dim=1)


                logging.debug("goodness for prediction" + " " +
                      str(label) + ": " + str(goodness))
                all_labels_goodness.append(goodness)

            all_labels_goodness = torch.stack(all_labels_goodness, dim=1)

            # select the label with the maximum goodness
            predicted_labels = torch.argmax(all_labels_goodness, dim=1)
            logging.debug("predicted labels: " + str(predicted_labels))
            logging.debug("actual labels: " + str(labels))

            total = data.size(0)
            correct = (predicted_labels == labels).sum().item()

        accuracy = 100 * correct / total
        logging.info(f'test accuracy: {accuracy}%')

        return accuracy

    def __advance_layers_train(self, input_data, label_data, should_damp):
        """
        Advances the training process for all layers in the network by computing
        the loss for each layer and updating their activations. 

        The method handles different layer scenarios: if it's a single layer,
        both the input data and label data are used for training. If it's the
        first or last layer in a multi-layer configuration, only the input data
        or label data is used, respectively. For layers in the middle of a
        multi-layer network, neither the input data nor the label data is used.

        Args:
            input_data (torch.Tensor): The input data for the network.

            label_data (torch.Tensor): The target labels for the network.

            should_damp (bool): A flag to determine whether the activation
            damping should be applied during training.

        Returns:
            total_loss (float): The accumulated loss over all layers in the
            network during the current training step.

        Note:
            The layer's 'train' method is expected to return a loss value, which
            is accumulated to compute the total loss for the network. After
            training each layer, their stored activations are advanced by
            calling the 'advance_stored_activations' method.
        """
        total_loss = 0
        for i, layer in enumerate(self.layers):
            logging.debug("Training layer " + str(i))
            loss = None
            if i == 0 and len(self.layers) == 1:
                loss = layer.train(self.optimizer, input_data,
                                   label_data, should_damp)
            elif i == 0:
                loss = layer.train(
                    self.optimizer, input_data, None, should_damp)
            elif i == len(self.layers) - 1:
                loss = layer.train(self.optimizer, None,
                                   label_data, should_damp)
            else:
                loss = layer.train(self.optimizer, None, None, should_damp)
            total_loss += loss
            logging.info("Loss for layer " + str(i) + ": " + str(loss))

        logging.debug("Trained activations for layer " +
                      str(i))

        for layer in self.layers:
            layer.advance_stored_activations()

        return total_loss

    def __advance_layers_forward(self, mode, input_data, label_data, should_damp):
        """
        Executes a forward pass through all layers of the network using the
        given mode, input data, label data, and a damping flag.

        The method handles different layer scenarios: if it's a single layer,
        both the input data and label data are used for the forward pass. If
        it's the first or last layer in a multi-layer configuration, only the
        input data or label data is used, respectively. For layers in the middle
        of a multi-layer network, neither the input data nor the label data is
        used.

        After the forward pass, the method advances the stored activations for
        all layers.

        Args:
            mode (ForwardMode): An enum representing the mode of forward
            propagation. This could be PositiveData, NegativeData, or
            PredictData. input_data

            (torch.Tensor): The input data for the
            network.
            
            label_data (torch.Tensor): The target labels for the
            network.
            
            should_damp (bool): A flag to determine whether the
            activation damping should be applied during the forward pass.

        Note:
            This method doesn't return any value. It modifies the internal state
            of the layers by performing a forward pass and advancing their
            stored activations.
        """
        for i, layer in enumerate(self.layers):
            if i == 0 and len(self.layers) == 1:
                layer.forward(mode, input_data, label_data, should_damp)
            elif i == 0:
                layer.forward(mode, input_data, None, should_damp)
            elif i == len(self.layers) - 1:
                layer.forward(mode, None, label_data, should_damp)
            else:
                layer.forward(mode, None, None, should_damp)

        for layer in self.layers:
            layer.advance_stored_activations()


class HiddenLayer(nn.Module):
    """
    A HiddenLayer class for a novel Forward-Forward Recurrent Network, with
    inspiration drawn from Boltzmann Machines and Noise Contrastive Estimation.
    This network design is characterized by two distinct forward passes, each
    with specific objectives: one is dedicated to processing positive ("real")
    data with the aim of enhancing the 'goodness' across every hidden layer,
    while the other is tasked with processing negative data and adjusting the
    weights to reduce the 'goodness' metric. 

    The HiddenLayer is essentially a node within this network, with possible
    connections to both preceding and succeeding layers, depending on its
    specific location within the network architecture. The first layer in this
    setup is connected directly to the input data, and the last layer maintains
    a connection to the output data. The intermediate layers establish a link to
    both their previous and next layers, if available. 

    In each HiddenLayer, a forward linear transformation and a backward linear
    transformation are defined. The forward transformation is applied to the
    activations from the previous layer, while the backward transformation is
    applied to the activations of the subsequent layer. The forward
    transformation helps in propagating the data through the network, and the
    backward transformation is key in the learning process where it aids in the
    adjustment of weights based on the output or next layer's activations.
    """
    def __init__(self, train_batch_size, test_batch_size, prev_size, size, damping_factor):
        super(HiddenLayer, self).__init__()

        self.train_activations_dim = (train_batch_size, size)
        self.test_activations_dim = (test_batch_size, size)

        self.damping_factor = damping_factor

        self.pos_activations = None
        self.neg_activations = None
        self.predict_activations = None
        self.reset_activations(True)

        # TODO: weight ordering?
        self.forward_linear = nn.Linear(prev_size, size)
        self.backward_linear = nn.Linear(size, prev_size)

        self.previous_layer = None
        self.next_layer = None

    def _apply(self, fn):
        """
        Override apply, but we don't want to apply to sibling layers as that
        will cause a stack overflow. The hidden layers are contained in a
        collection in the higher-level RecurrentFFNet. They will all get the
        apply call from there.
        """
        # Apply `fn` to each parameter and buffer of this layer
        for param in self._parameters.values():
            if param is not None:
                # Tensors stored in modules are graph leaves, and we don't
                # want to create copy nodes, so we have to unpack the data.
                param.data = fn(param.data)
                if param._grad is not None:
                    param._grad.data = fn(param._grad.data)

        for key, buf in self._buffers.items():
            if buf is not None:
                self._buffers[key] = fn(buf)

        # Then remove `previous_layer` and `next_layer` temporarily
        previous_layer = self.previous_layer
        next_layer = self.next_layer
        self.previous_layer = None
        self.next_layer = None

        # Apply `fn` to submodules
        for module in self.children():
            module._apply(fn)

        # Restore `previous_layer` and `next_layer`
        self.previous_layer = previous_layer
        self.next_layer = next_layer

        return self

    # TODO: we can optimize for memory by enabling distinct train / predict mode
    # TODO: fix hacky isTraining flag
    def reset_activations(self, isTraining):
        activations_dim = None
        if isTraining:
            activations_dim = self.train_activations_dim
        else:
            activations_dim = self.test_activations_dim

        pos_activations_current = torch.zeros(
            activations_dim[0], activations_dim[1]).to(device)
        pos_activations_previous = torch.zeros(
            activations_dim[0], activations_dim[1]).to(device)
        self.pos_activations = Activations(
            pos_activations_current, pos_activations_previous)

        neg_activations_current = torch.zeros(
            activations_dim[0], activations_dim[1]).to(device)
        neg_activations_previous = torch.zeros(
            activations_dim[0], activations_dim[1]).to(device)
        self.neg_activations = Activations(
            neg_activations_current, neg_activations_previous)

        predict_activations_current = torch.zeros(
            activations_dim[0], activations_dim[1]).to(device)
        predict_activations_previous = torch.zeros(
            activations_dim[0], activations_dim[1]).to(device)
        self.predict_activations = Activations(
            predict_activations_current, predict_activations_previous)

    def advance_stored_activations(self):
        self.pos_activations.advance()
        self.neg_activations.advance()
        self.predict_activations.advance()

    def set_previous_layer(self, previous_layer):
        self.previous_layer = previous_layer

    def set_next_layer(self, next_layer):
        self.next_layer = next_layer

    def train(self, optimizer, input_data, label_data, should_damp):
        optimizer.zero_grad()

        pos_activations = None
        neg_activations = None
        if input_data != None and label_data != None:
            (pos_input, neg_input) = input_data
            (pos_labels, neg_labels) = label_data
            pos_activations = self.forward(
                ForwardMode.PositiveData, pos_input, pos_labels, should_damp)
            neg_activations = self.forward(
                ForwardMode.NegativeData, neg_input, neg_labels, should_damp)
        elif input_data != None:
            (pos_input, neg_input) = input_data
            pos_activations = self.forward(
                ForwardMode.PositiveData, pos_input, None, should_damp)
            neg_activations = self.forward(
                ForwardMode.NegativeData, neg_input, None, should_damp)
        elif label_data != None:
            (pos_labels, neg_labels) = label_data
            pos_activations = self.forward(
                ForwardMode.PositiveData, None, pos_labels, should_damp)
            neg_activations = self.forward(
                ForwardMode.NegativeData, None, neg_labels, should_damp)
        else:
            pos_activations = self.forward(
                ForwardMode.PositiveData, None, None, should_damp)
            neg_activations = self.forward(
                ForwardMode.NegativeData, None, None, should_damp)

        pos_goodness = layer_activations_to_goodness(pos_activations)
        neg_goodness = layer_activations_to_goodness(neg_activations)

        logging.debug("pos goodness: " + str(pos_goodness))
        logging.debug("neg goodness: " + str(neg_goodness))

        # Loss function equivelent to:
        # z = log(1 + exp(((-x + 2) + (y - 2))/2)
        layer_loss = torch.log(1 + torch.exp(torch.cat([
            (-1 * pos_goodness) + THRESHOLD,
            neg_goodness - THRESHOLD
        ]))).mean()

        layer_loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        return layer_loss

    def forward(self, mode, data, labels, should_damp):
        """
        Propagates input data forward through the network, updating the
        activation state of the current layer based on the operating mode. 

        Handles various scenarios depending on the layer configuration in the
        network (input layer, output layer, or a middle layer).

        Args:
            mode (ForwardMode enum): Indicates the type of data being propagated
            (positive, negative, or prediction).

            data (torch.Tensor or None): The input data for the layer. If
            `None`, it indicates that this layer is not the input layer.

            labels (torch.Tensor or None): The target labels for the layer. If
            `None`, it indicates that this layer is not the output layer.
            
            should_damp (bool): A flag to determine whether the activation
            damping should be applied.

        Returns:
            new_activation (torch.Tensor): The updated activation state of the
            layer after the forward propagation.

        Note:
            'Damping' here refers to a technique used to smoothen the changes in
            the layer activations over time. In this function, damping is
            implemented as a weighted average of the previous and the newly
            computed activations, controlled by the `self.damping_factor`.
            
            The function expects to receive input data and/or labels depending
            on the layer. The absence of both implies the current layer is a
            'middle' layer. If only the labels are missing, this layer is an
            'input' layer, while if only the data is missing, it's an 'output'
            layer. If both are provided, the network has only a single layer.

            All four scenarios are handled separately in the function, although
            the general procedure is similar: compute new activations based on
            the received inputs (and possibly, depending on the layer's
            position, the activations of the adjacent layers), optionally apply
            damping, update the current layer's activations, and return the new
            activations.
        """
        # Make sure assumptions aren't violated regarding layer connectivity.
        if data == None:
            assert self.previous_layer != None
        if labels == None:
            assert self.next_layer != None

        # Middle layer.
        if data == None and labels == None:
            next_layer_prev_timestep_activations = None
            prev_layer_prev_timestep_activations = None
            prev_act = None
            if mode == ForwardMode.PositiveData:
                next_layer_prev_timestep_activations = self.next_layer.pos_activations.previous
                prev_layer_prev_timestep_activations = self.previous_layer.pos_activations.previous
                prev_act = self.pos_activations.previous
            elif mode == ForwardMode.NegativeData:
                next_layer_prev_timestep_activations = self.next_layer.neg_activations.previous
                prev_layer_prev_timestep_activations = self.previous_layer.neg_activations.previous
                prev_act = self.neg_activations.previous
            elif mode == ForwardMode.PredictData:
                next_layer_prev_timestep_activations = self.next_layer.predict_activations.previous
                prev_layer_prev_timestep_activations = self.previous_layer.predict_activations.previous
                prev_act = self.predict_activations.previous
            next_layer_prev_timestep_activations = next_layer_prev_timestep_activations.detach()
            prev_layer_prev_timestep_activations = prev_layer_prev_timestep_activations.detach()
            prev_act = prev_act.detach()

            # Compute mean and standard deviation for next_layer
            next_layer_mean = next_layer_prev_timestep_activations.mean(dim=1, keepdim=True)
            next_layer_std = next_layer_prev_timestep_activations.std(dim=1, keepdim=True)

            # Compute mean and standard deviation for prev_layer
            prev_layer_mean = prev_layer_prev_timestep_activations.mean(dim=1, keepdim=True)
            prev_layer_std = prev_layer_prev_timestep_activations.std(dim=1, keepdim=True)

            # Apply standardization
            next_layer_stdized = (next_layer_prev_timestep_activations - next_layer_mean) / (next_layer_std + EPSILON)
            prev_layer_stdized = (prev_layer_prev_timestep_activations - prev_layer_mean) / (prev_layer_std + EPSILON)

            new_activation = F.relu(F.linear(prev_layer_stdized, self.forward_linear.weight) +
                                    F.linear(next_layer_stdized,
                                             self.next_layer.backward_linear.weight))
            if should_damp:
                old_activation = new_activation
                new_activation = (1 - self.damping_factor) * \
                    prev_act + self.damping_factor * old_activation

            if mode == ForwardMode.PositiveData:
                self.pos_activations.current = new_activation
            elif mode == ForwardMode.NegativeData:
                self.neg_activations.current = new_activation
            elif mode == ForwardMode.PredictData:
                self.predict_activations.current = new_activation

            return new_activation

        # Single layer scenario. Hidden layer connected to input layer and
        # output layer.
        elif data != None and labels != None:
            prev_act = None
            if mode == ForwardMode.PositiveData:
                prev_act = self.pos_activations.previous
            elif mode == ForwardMode.NegativeData:
                prev_act = self.neg_activations.previous
            elif mode == ForwardMode.PredictData:
                prev_act = self.predict_activations.previous
            prev_act = prev_act.detach()

            new_activation = F.relu(F.linear(
                data, self.forward_linear.weight) + F.linear(labels, self.next_layer.backward_linear.weight))

            if should_damp:
                old_activation = new_activation
                new_activation = (1 - self.damping_factor) * \
                    prev_act + self.damping_factor * old_activation

            if mode == ForwardMode.PositiveData:
                self.pos_activations.current = new_activation
            elif mode == ForwardMode.NegativeData:
                self.neg_activations.current = new_activation
            elif mode == ForwardMode.PredictData:
                self.predict_activations.current = new_activation

            return new_activation

        # Input layer scenario. Connected to input layer and hidden layer.
        elif data != None:
            prev_act = None
            next_layer_prev_timestep_activations = None
            if mode == ForwardMode.PositiveData:
                next_layer_prev_timestep_activations = self.next_layer.pos_activations.previous
                prev_act = self.pos_activations.previous
            elif mode == ForwardMode.NegativeData:
                next_layer_prev_timestep_activations = self.next_layer.neg_activations.previous
                prev_act = self.neg_activations.previous
            elif mode == ForwardMode.PredictData:
                next_layer_prev_timestep_activations = self.next_layer.predict_activations.previous
                prev_act = self.predict_activations.previous
            prev_act = prev_act.detach()
            next_layer_prev_timestep_activations = next_layer_prev_timestep_activations.detach()

            # Compute mean and standard deviation for next_layer
            next_layer_mean = next_layer_prev_timestep_activations.mean(dim=1, keepdim=True)
            next_layer_std = next_layer_prev_timestep_activations.std(dim=1, keepdim=True)

            # Apply standardization
            next_layer_stdized = (next_layer_prev_timestep_activations - next_layer_mean) / (next_layer_std + EPSILON)

            new_activation = F.relu(F.linear(data, self.forward_linear.weight) + F.linear(
                next_layer_stdized, self.next_layer.backward_linear.weight))

            if should_damp:
                old_activation = new_activation
                new_activation = (1 - self.damping_factor) * \
                    prev_act + self.damping_factor * old_activation

            if mode == ForwardMode.PositiveData:
                self.pos_activations.current = new_activation
            elif mode == ForwardMode.NegativeData:
                self.neg_activations.current = new_activation
            elif mode == ForwardMode.PredictData:
                self.predict_activations.current = new_activation

            return new_activation

        # Output layer scenario. Connected to hidden layer and output layer.
        elif labels != None:
            prev_layer_prev_timestep_activations = None
            prev_act = None
            if mode == ForwardMode.PositiveData:
                prev_layer_prev_timestep_activations = self.previous_layer.pos_activations.previous
                prev_act = self.pos_activations.previous
            elif mode == ForwardMode.NegativeData:
                prev_layer_prev_timestep_activations = self.previous_layer.neg_activations.previous
                prev_act = self.neg_activations.previous
            elif mode == ForwardMode.PredictData:
                prev_layer_prev_timestep_activations = self.previous_layer.predict_activations.previous
                prev_act = self.predict_activations.previous
            prev_act = prev_act.detach()
            prev_layer_prev_timestep_activations = prev_layer_prev_timestep_activations.detach()

            # Compute mean and standard deviation for prev_layer
            prev_layer_mean = prev_layer_prev_timestep_activations.mean(dim=1, keepdim=True)
            prev_layer_std = prev_layer_prev_timestep_activations.std(dim=1, keepdim=True)

            # Apply standardization
            prev_layer_stdized = (prev_layer_prev_timestep_activations - prev_layer_mean) / (prev_layer_std + EPSILON)

            new_activation = F.relu(F.linear(prev_layer_stdized,
                                             self.forward_linear.weight) + F.linear(labels, self.next_layer.backward_linear.weight))

            if should_damp:
                old_activation = new_activation
                new_activation = (1 - self.damping_factor) * \
                    prev_act + self.damping_factor * old_activation

            if mode == ForwardMode.PositiveData:
                self.pos_activations.current = new_activation
            elif mode == ForwardMode.NegativeData:
                self.neg_activations.current = new_activation
            elif mode == ForwardMode.PredictData:
                self.predict_activations.current = new_activation

            return new_activation


if __name__ == "__main__":
    # This needs to change if not running mps backend
    assert (torch.backends.mps.is_available())
    assert (torch.backends.mps.is_built())
    device = torch.device("mps")

    # Pytorch utils.
    torch.autograd.set_detect_anomaly(True)
    torch.manual_seed(1234)

    wandb.init(
        # set the wandb project where this run will be logged
        project="Recurrent-FF",
        
        # track hyperparameters and run metadata
        config={
        "architecture": "Recurrent-FF",
        "dataset": "MNIST",
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "layers": str(LAYERS),
        "iterations": ITERATIONS,
        "threshold": THRESHOLD,
        "damping_factor": DAMPING_FACTOR,
        "epsilon": EPSILON,
        }
    )

    # Generate train data.
    train_loader, test_loader = MNIST_loaders()
    # TODO: put this on the device initially?
    x, y_pos = next(iter(train_loader))
    x, y_pos = x.to(device), y_pos.to(device)
    train_batch_size = len(x)

    shuffled_labels = torch.randperm(x.size(0))
    y_neg = y_pos[shuffled_labels]

    positive_one_hot_labels = torch.zeros(
        len(y_pos), NUM_CLASSES, device=device)
    positive_one_hot_labels.scatter_(1, y_pos.unsqueeze(1), 1.0)

    negative_one_hot_labels = torch.zeros(
        len(y_neg), NUM_CLASSES, device=device)
    negative_one_hot_labels.scatter_(1, y_neg.unsqueeze(1), 1.0)

    input_data = InputData(x, x)
    label_data = LabelData(positive_one_hot_labels, negative_one_hot_labels)

    # Generate test data.
    x, y = next(iter(test_loader))
    x, y = x.to(device), y.to(device)
    test_batch_size = len(x)
    labels = y
    one_hot_labels = torch.zeros(
        len(y), NUM_CLASSES, device=device)
    one_hot_labels.scatter_(1, y.unsqueeze(1), 1.0)
    test_data = TestData(x, one_hot_labels, labels)

    # Create and run model.
    model = RecurrentFFNet(train_batch_size, test_batch_size, INPUT_SIZE, LAYERS, NUM_CLASSES).to(device)

    model.train(input_data, label_data, test_data)

    model.predict(test_data)
