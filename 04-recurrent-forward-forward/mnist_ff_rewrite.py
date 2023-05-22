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
from enum import Enum
import numpy as np

# logging.basicConfig(filename="log.log", level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# QUESTIONS:
# 1 - First layer activations get very sparse?
# 2 - Why is the validation goodness so much lower than the training goodness?
# TODO:
# 1 - Implement side connections as shown in Fig3
# 3 - Implement dedicated recurrent weights
# 4 - Experiment with more layers
# 5 - Experiment with weight initialization
# 6 - Experiment with activation initialization
# 7 - Docstrings
# 8 - Stop using nn.Linear if all we need are simple weights
# 9 - Try different optimizers
# 10 - Model.eval()? We are not using it now.
# 11 - Why are we getting an error?
# 12 - Experiment with more optimizers
# 13 - cap loss offset from sufficiently learned examples (we shouldn't be seeing a goodness of like 7 for pos, it is too high and allowing other parts of the network to suffer)

EPOCHS = 1000
ITERATIONS = 10
THRESHOLD = 1
LEARNING_RATE = 0.0001
DAMPING_FACTOR = 0.7
# LEARNING_RATE = 0.0001

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
# TODO: no more requires_grad
def activations_to_goodness(activations):
    goodness = []
    for act in activations:
        goodness_for_layer = torch.mean(
            torch.square(act), dim=1).requires_grad_()
        goodness.append(goodness_for_layer)

    return goodness


def layer_activations_to_goodness(layer_activations):
    goodness_for_layer = torch.mean(
        torch.square(layer_activations), dim=1).requires_grad_()

    return goodness_for_layer


class RecurrentFFNet(nn.Module):
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
        for epoch in range(0, EPOCHS):
            logging.info("Epoch: " + str(epoch))
            self.reset_activations(True)

            for preinit_step in range(0, len(self.layers)):
                logging.info("Preinitialization step: " + str(preinit_step))
                self.__advance_layers_forward(ForwardMode.PositiveData,
                                              input_data.pos_input, label_data.pos_labels, False)
                self.__advance_layers_forward(ForwardMode.NegativeData,
                                              input_data.neg_input, label_data.neg_labels, False)
                for layer in self.layers:
                    layer.advance_stored_activations()

                # for layer in self.layers:
                #     print("layer activations previous: " +
                #           str(layer.pos_activations.previous))
                #     print("layer activations current: " +
                #           str(layer.pos_activations.current))

            for iteration in range(0, ITERATIONS):
                logging.info("Iteration: " + str(iteration))
                total_loss = self.__advance_layers_train(
                    input_data, label_data, True)
                logging.info("Average layer loss: " +
                             str(total_loss / len(self.layers)))

            self.predict(test_data)

    def predict(self, test_data):
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

                for preinit in range(0, len(self.layers)):
                    self.__advance_layers_forward(ForwardMode.PredictData,
                                                  data, one_hot_labels, False)

                    # TODO: this can be refactored into advance layers forwards
                    for layer in self.layers:
                        layer.advance_stored_activations()

                for iteration in range(0, ITERATIONS):
                    self.__advance_layers_forward(ForwardMode.PredictData,
                                                  data, one_hot_labels, True)

                # TODO: optimize this to not have lists
                activations = [
                    layer.predict_activations.current for layer in self.layers]
                # print("-----activations: " + str(activations))
                goodness = activations_to_goodness(activations)
                # TODO: convert to DEBUG?
                print("layer goodness for prediction" + " " +
                      str(label) + ": " + str(goodness))
                goodness = torch.stack(goodness, dim=1).mean(dim=1)
                all_labels_goodness.append(goodness)
                print("overall goodness for prediction" +
                      str(label) + ": " + str(goodness))

            all_labels_goodness = torch.stack(all_labels_goodness, dim=1)

            # select the label with the maximum goodness
            predicted_labels = torch.argmax(all_labels_goodness, dim=1)
            print("predicted labels: " + str(predicted_labels))
            print("actual labels: " + str(labels))

            total = data.size(0)
            correct = (predicted_labels == labels).sum().item()

        accuracy = 100 * correct / total
        logging.info(f'test accuracy: {accuracy}%')

        return accuracy

    def __advance_layers_train(self, input_data, label_data, should_damp):
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

        # TODO: comment why this is needed here but we have forward in train
        for layer in self.layers:
            layer.advance_stored_activations()

        return total_loss

    def __advance_layers_forward(self, mode, input_data, label_data, should_damp):
        for i, layer in enumerate(self.layers):
            if i == 0 and len(self.layers) == 1:
                layer.forward(mode, input_data, label_data, should_damp)
            elif i == 0:
                layer.forward(mode, input_data, None, should_damp)
            elif i == len(self.layers) - 1:
                layer.forward(mode, None, label_data, should_damp)
            else:
                layer.forward(mode, None, None, should_damp)


class HiddenLayer(nn.Module):
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

        # print("layer activations previous: " +
        #       str(self.pos_activations.previous))
        # print("layer activations current: " +
        #       str(self.pos_activations.current))

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

        # print("----positive activations: " +
        #       str(pos_activations.cpu().detach().numpy()))
        # print("----negative activations: " +
        #       str(neg_activations.cpu().detach().numpy()))

        # print("-----positive activations:")
        # for i in range(pos_activations.shape[0]):
        #     for j in range(pos_activations.shape[1]):
        #         print(pos_activations[i][j].item(), end=' ')
        #     print()
        # print()

        # print("-----negative activations:")
        # for i in range(pos_activations.shape[0]):
        #     for j in range(pos_activations.shape[1]):
        #         print(pos_activations[i][j].item(), end=' ')
        #     print()
        # print()

        pos_goodness = layer_activations_to_goodness(pos_activations)
        neg_goodness = layer_activations_to_goodness(neg_activations)

        logging.debug("pos goodness: " + str(pos_goodness))
        logging.debug("neg goodness: " + str(neg_goodness))

        layer_loss = torch.log(1 + torch.exp(torch.cat([
            (-1 * pos_goodness) + THRESHOLD,
            neg_goodness - THRESHOLD
        ]))).mean()

        layer_loss.backward()

        # torchviz.make_dot(layer_loss, params=dict(
        #     model.named_parameters())).render("graph", format="png")

        # print("unclipped grads")
        # if input_data != None:
        #     print("[expect] first layer grad (forwards): " +
        #           str(self.forward_linear.weight.grad))
        #     print("[expect] first layer grad (backwards): " +
        #           str(self.next_layer.backward_linear.weight.grad))
        #     print("second layer grad (forwards): " +
        #           str(self.next_layer.forward_linear.weight.grad))
        #     print("second layer grad (backwards): " +
        #           str(self.next_layer.next_layer.backward_linear.weight.grad))
        # else:
        #     print("first layer grad (forwards): " +
        #           str(self.previous_layer.forward_linear.weight.grad))
        #     print("first layer grad (backwards): " +
        #           str(self.backward_linear.weight.grad))
        #     print("[expect] second layer grad (forwards): " +
        #           str(self.forward_linear.weight.grad))
        #     print("[expect] second layer grad (backwards): " +
        #           str(self.next_layer.backward_linear.weight.grad))

        # input()

        # torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1)

        # print("clipped grads")
        # if input_data != None:
        #     print("[expect] first layer grad (forwards): " +
        #           str(self.forward_linear.weight.grad))
        #     print("[expect] first layer grad (backwards): " +
        #           str(self.next_layer.backward_linear.weight.grad))
        #     print("second layer grad (forwards): " +
        #           str(self.next_layer.forward_linear.weight.grad))
        #     print("second layer grad (backwards): " +
        #           str(self.next_layer.next_layer.backward_linear.weight.grad))
        # else:
        #     print("first layer grad (forwards): " +
        #           str(self.previous_layer.forward_linear.weight.grad))
        #     print("first layer grad (backwards): " +
        #           str(self.backward_linear.weight.grad))
        #     print("[expect] second layer grad (forwards): " +
        #           str(self.forward_linear.weight.grad))
        #     print("[expect] second layer grad (backwards): " +
        #           str(self.next_layer.backward_linear.weight.grad))

        optimizer.step()
        optimizer.zero_grad()

        return layer_loss

    def forward(self, mode, data, labels, should_damp):
        # print("layer activations previous: " +
        #       str(self.pos_activations.previous))
        # print("layer activations current: " +
        #       str(self.pos_activations.current))

        if data == None:
            assert self.previous_layer != None
        if labels == None:
            assert self.next_layer != None

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

            next_layer_norm = next_layer_prev_timestep_activations.norm(
                p=2, dim=1, keepdim=True)
            prev_layer_norm = prev_layer_prev_timestep_activations.norm(
                p=2, dim=1, keepdim=True)
            new_activation = F.relu(F.linear(prev_layer_norm, self.forward_linear.weight) +
                                    F.linear(next_layer_norm,
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

        elif data != None:
            # print("-----problematic activation case here")
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

            next_layer_norm = next_layer_prev_timestep_activations.norm(
                p=2, dim=1, keepdim=True)

            new_activation = F.relu(F.linear(data, self.forward_linear.weight) + F.linear(
                next_layer_prev_timestep_activations, self.next_layer.backward_linear.weight))

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

            # print("new activations: ", new_activation)
            return new_activation

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

            prev_layer_norm = prev_layer_prev_timestep_activations.norm(
                p=2, dim=1, keepdim=True)

            new_activation = F.relu(F.linear(prev_layer_prev_timestep_activations,
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
    model = RecurrentFFNet(train_batch_size, test_batch_size, INPUT_SIZE, [
        500, 250], NUM_CLASSES).to(device)

    model.train(input_data, label_data, test_data)

    model.predict(test_data)
