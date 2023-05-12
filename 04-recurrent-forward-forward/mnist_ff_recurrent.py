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

        # Define layer normalization for hidden layers
        # self.layer_norms = nn.ModuleList(
        #     [nn.LayerNorm(size) for size in hidden_sizes])

    def forward_timestep_test(self, input, prev_activations, labels, should_damp=True):
        prev_activations_norm = []
        for i in range(0, len(prev_activations)):
            prev_activations[i] = prev_activations[i].detach()

            if torch.all(prev_activations[i] == 0):
                prev_activations_norm.append(prev_activations[i])
                continue

            norm = prev_activations[i].norm(p=2, dim=1, keepdim=True)
            prev_activations_norm.append(prev_activations[i].div(norm))

        new_activations = []
        output_layer_weights = self.layers[-1].weight.t()
        if len(prev_activations) == 1:
            pos_new_act = F.linear(
                labels, output_layer_weights) + F.linear(input, self.layers[0].weight)
            if should_damp:
                # TODO: add note
                pos_new_act = (1 - self.damping_factor) * prev_activations[0] + \
                    self.damping_factor * pos_new_act
            new_activations.append(F.relu(pos_new_act))
        else:
            for i, (prev_act, prev_act_norm, layer) in enumerate(zip(prev_activations, prev_activations_norm, self.layers[:-1])):
                if i == 0:
                    new_activations.append(F.linear(input, layer.weight) + F.linear(
                        prev_activations_norm[i + 1], self.layers[i+1].weight.t()))
                elif i == len(prev_activations) - 1:
                    new_activations.append(F.linear(prev_activations_norm[i - 1], layer.weight) + F.linear(
                        labels, output_layer_weights))
                else:
                    new_activations.append(F.linear(prev_act_norm[i - 1], layer.weight) + F.linear(
                        prev_act_norm[i + 1], self.layers[i+1].weight.t()))

                if should_damp:
                    new_activations[i] = (1 - self.damping_factor) * prev_act + \
                        self.damping_factor * new_activations[i]

                new_activations[i] = F.relu(new_activations[i])

        return new_activations

    # TODO: implement side connections as shown in Fig3
    def forward_timestep_train(self, pos_input, pos_prev_activations, pos_labels, neg_input, neg_prev_activations, neg_labels, optimizer, should_train_and_damp=True):
        pos_prev_activations_norm = []
        for i in range(0, len(pos_prev_activations)):
            pos_prev_activations[i] = pos_prev_activations[i].detach()

            if torch.all(pos_prev_activations[i] == 0):
                pos_prev_activations_norm.append(pos_prev_activations[i])
                continue

            norm = pos_prev_activations[i].norm(p=2, dim=1, keepdim=True)
            pos_prev_activations_norm.append(pos_prev_activations[i].div(norm))

        neg_prev_activations_norm = []
        for i in range(0, len(neg_prev_activations)):
            neg_prev_activations[i] = neg_prev_activations[i].detach()

            if torch.all(neg_prev_activations[i] == 0):
                neg_prev_activations_norm.append(neg_prev_activations[i])
                continue

            norm = neg_prev_activations[i].norm(p=2, dim=1, keepdim=True)
            neg_prev_activations_norm.append(neg_prev_activations[i].div(norm))

        output_layer_weights = self.layers[-1].weight.t()
        pos_new_activations = []
        neg_new_activations = []
        layer_losses = []
        # TODO: check needs cleanup
        if len(pos_prev_activations_norm) == 1:
            optimizer.zero_grad()
            pos_new_act = F.linear(
                pos_labels, output_layer_weights) + F.linear(pos_input, self.layers[0].weight)
            if should_train_and_damp:
                # TODO: add note
                pos_new_act = (1 - self.damping_factor) * pos_prev_activations[0] + \
                    self.damping_factor * pos_new_act
            pos_new_activations.append(F.relu(pos_new_act))

            neg_new_act = F.linear(
                neg_labels, output_layer_weights) + F.linear(neg_input, self.layers[0].weight)
            # TODO: this wasn't tested with single layer bench, so need to test again
            if should_train_and_damp:
                # TODO: add note
                neg_new_act = (1 - self.damping_factor) * neg_prev_activations[0] + \
                    self.damping_factor * neg_new_act
            neg_new_activations.append(F.relu(neg_new_act))

            if should_train_and_damp:
                # print(pos_new_activations[0])
                pos_goodness = layer_activations_to_goodness(
                    pos_new_activations[0])
                neg_goodness = layer_activations_to_goodness(
                    neg_new_activations[0])

                print("LAYER 0 positive goodness: ", pos_goodness)
                print("LAYER 0 negative goodness: ", neg_goodness)

                layer_loss = torch.log(1 + torch.exp(torch.cat([
                    (-1 * pos_goodness) + THRESHOLD,
                    neg_goodness - THRESHOLD
                ]))).mean()
                layer_losses.append(layer_loss)
                layer_loss.backward()
                # torchviz.make_dot(layer_loss, params=dict(
                #     model.named_parameters())).render("graph", format="png")
                optimizer.step()

        else:
            for i, (pos_prev_act, neg_prev_act, pos_prev_act_norm, neg_prev_act_norm, layer) in enumerate(zip(pos_prev_activations, neg_prev_activations, pos_prev_activations_norm, neg_prev_activations_norm, self.layers[:-1])):
                optimizer.zero_grad()
                # first layer gets the input image
                if i == 0:
                    # print("------, ", i)
                    # print("first weights")
                    # print(input_image.shape)
                    # print(layer.weight.shape)

                    # print("second")
                    # print(prev_norm_act[i + 1].shape)
                    # print(self.layers[i+1].weight.t().shape)

                    pos_new_activations.append(F.linear(pos_input, layer.weight) + F.linear(
                        pos_prev_activations_norm[i + 1], self.layers[i+1].weight.t()))

                    neg_new_activations.append(F.linear(neg_input, layer.weight) + F.linear(
                        neg_prev_activations_norm[i + 1], self.layers[i+1].weight.t()))
                # last hidden layer gets the previous layer's activation and the one-hot labels
                # TODO: check needs to be refactored
                elif i == len(pos_prev_activations) - 1:
                    # print("------, ", i)
                    # print("first weights")
                    # print(prev_norm_act[i - 1].shape)
                    # print(layer.weight.shape)
                    # F.linear(prev_norm_act[i - 1], layer.weight)

                    # print("second")
                    # print(one_hot_labels.shape)
                    # print(output_layer_weights.shape)
                    # F.linear(one_hot_labels, output_layer_weights)

                    pos_new_activations.append(F.linear(pos_prev_activations_norm[i - 1], layer.weight) + F.linear(
                        pos_labels, output_layer_weights))
                    neg_new_activations.append(F.linear(neg_prev_activations_norm[i - 1], layer.weight) + F.linear(
                        neg_labels, output_layer_weights))
                # other layers get activations from the layers above and below
                else:
                    pos_new_activations.append(F.linear(pos_prev_act_norm[i - 1], layer.weight) + F.linear(
                        pos_prev_act_norm[i + 1], self.layers[i+1].weight.t()))
                    neg_new_activations.append(F.linear(neg_prev_act_norm[i - 1], layer.weight) + F.linear(
                        neg_prev_act_norm[i + 1], self.layers[i+1].weight.t()))

                if should_train_and_damp:
                    pos_new_activations[i] = (1 - self.damping_factor) * pos_prev_act + \
                        self.damping_factor * pos_new_activations[i]
                    neg_new_activations[i] = (1 - self.damping_factor) * neg_prev_act + \
                        self.damping_factor * neg_new_activations[i]

                pos_new_activations[i] = F.relu(pos_new_activations[i])
                neg_new_activations[i] = F.relu(neg_new_activations[i])

                if should_train_and_damp:
                    pos_goodness = layer_activations_to_goodness(
                        pos_new_activations[i])
                    neg_goodness = layer_activations_to_goodness(
                        neg_new_activations[i])

                    print("LAYER " + str(i) + " positive goodness: " + str(pos_goodness))
                    print("LAYER " + str(i) + " negative goodness: " + str(neg_goodness))

                    layer_loss = torch.log(1 + torch.exp(torch.cat([
                        (-1 * pos_goodness) + THRESHOLD,
                        neg_goodness - THRESHOLD
                    ]))).mean()
                    layer_losses.append(layer_loss)
                    layer_loss.backward()
                    # torchviz.make_dot(layer_loss, params=dict(
                    #     model.named_parameters())).render("graph", format="png")
                    # input()
                    optimizer.step()

        return pos_new_activations, neg_new_activations, layer_losses


def layer_activations_to_goodness(layer_activations):
    goodness_for_layer = torch.mean(
        torch.square(layer_activations), dim=1).requires_grad_()
    # print("goodness for layer shape")
    # print(goodness_for_layer)

    # print("goodness")
    # print(goodness)
    return goodness_for_layer


def activations_to_goodness(activations):
    goodness = []
    for act in activations:
        goodness_for_layer = torch.mean(
            torch.square(act), dim=1).requires_grad_()
        # print("goodness for layer shape")
        # print(goodness_for_layer)
        goodness.append(goodness_for_layer)

    # print("goodness")
    # print(goodness)
    return goodness

# todo: rename images to inputs
# todo: consider adding a way to ignore layer training if activations haven't reached during start of timestep processing


def train(model, positive_images, positive_labels, negative_images, negative_labels, optimizer, device):
    model.train()

    # prepare positive one-hot encoded labels
    positive_one_hot_labels = torch.zeros(
        len(positive_labels), model.num_classes, device=device)
    positive_one_hot_labels.scatter_(1, positive_labels.unsqueeze(1), 1.0)

    # prepare negative one-hot encoded labels
    negative_one_hot_labels = torch.zeros(
        len(negative_labels), model.num_classes, device=device)
    negative_one_hot_labels.scatter_(1, negative_labels.unsqueeze(1), 1.0)

    # initialize activations with zeros
    # todo: dedup
    positive_activations = [torch.zeros(
        positive_images.shape[0], layer.out_features, device=device) for layer in model.layers[:-1]]
    negative_activations = [torch.zeros(
        negative_images.shape[0], layer.out_features, device=device) for layer in model.layers[:-1]]

    with torch.no_grad():
        for iteration in range(0, len(model.layers[:-1])):
            positive_activations, negative_activations, _layer_losses = model.forward_timestep_train(
                positive_images, positive_activations, positive_one_hot_labels, negative_images, negative_activations, negative_one_hot_labels, optimizer, False)

        # perform multiple iterations of the recurrent forward pass
    running_loss = 0
    for iteration in range(ITERATIONS):
        # print(torch.mps.current_allocated_memory() / (10 ** 9))

        positive_activations, negative_activations, layer_losses = model.forward_timestep_train(
            positive_images, positive_activations, positive_one_hot_labels, negative_images, negative_activations, negative_one_hot_labels, optimizer, True)

        # print("layer losses: ------- ", layer_losses)

        positive_goodness = activations_to_goodness(positive_activations)
        negative_goodness = activations_to_goodness(negative_activations)

        # print("goodness shape")
        # print(positive_goodness.shape)
        # print("positive goodness")
        # print(positive_goodness)

        # calculate negative goodness
        # print("negative goodness")
        # print(negative_goodness)

        # print(torch.mps.current_allocated_memory() / (10 ** 9))
        logging.info(
            f'iteration {iteration+1}, average layer loss: {sum(layer_losses) / len(layer_losses)}')

    return running_loss / ITERATIONS


def test(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            all_labels_goodness = []

            # evaluate goodness for each possible label
            for label in range(model.num_classes):
                one_hot_labels = torch.zeros(
                    images.shape[0], model.num_classes, device=device)
                one_hot_labels[:, label] = 1.0
                activations = [torch.zeros(
                    images.shape[0], layer.out_features, device=device) for layer in model.layers[:-1]]

                for iteration in range(0, len(model.layers[:-1])):
                    activations = model.forward_timestep_test(
                        images, activations, one_hot_labels, False)

                for _ in range(ITERATIONS):
                    activations = model.forward_timestep_test(
                        images, activations, one_hot_labels)

                goodness = activations_to_goodness(activations)
                goodness = torch.stack(goodness, dim=1).mean(dim=1)
                all_labels_goodness.append(goodness)

            # print("all labels goodness")
            # print(torch.stack(all_labels_goodness, dim=0).shape)
            # print(all_labels_goodness)

            all_labels_goodness = torch.stack(all_labels_goodness, dim=1)
            # print("all labels goodness shape")
            # print(all_labels_goodness)
            # print(all_labels_goodness.shape)

            # select the label with the maximum goodness
            predicted_labels = torch.argmax(all_labels_goodness, dim=1)
            print("predicted labels")
            # print(predicted_labels.shape)
            print(predicted_labels)
            print("labels")
            # print(labels.shape)
            print(labels)

            total += images.size(0)
            correct += (predicted_labels == labels).sum().item()

    accuracy = 100 * correct / total
    logging.info(f'test accuracy: {accuracy}%')

    return accuracy


if __name__ == "__main__":
    # this ensures that the current MacOS version is at least 12.3+
    assert (torch.backends.mps.is_available())
    # this ensures that the current current PyTorch installation was built with MPS activated.
    assert (torch.backends.mps.is_built())
    # set the output to device: mps
    device = torch.device("mps")

    torch.autograd.set_detect_anomaly(True)

    torch.manual_seed(1234)

    train_loader, test_loader = MNIST_loaders()

    model = RecurrentFFNet(784, [500, 250], 10).to(device)
    # TODO: decrease learning rate
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    num_epochs = 200

    for epoch in range(num_epochs):
        logging.info(f'Training starting...')

        x, y = next(iter(train_loader))
        x, y = x.to(device), y.to(device)

        x_pos = x
        y_pos = y

        shuffled_labels = torch.randperm(x.size(0))
        y_neg = y[shuffled_labels]
        x_neg = x

        loss = train(model, x_pos, y_pos, x_neg, y_neg, optimizer, device)
        logging.info(f'Epoch {epoch+1}, Loss: {loss}')

        accuracy = test(model, test_loader, device)
        logging.info(f'Epoch {epoch+1}, Loss: {loss}, Accuracy: {accuracy}')
