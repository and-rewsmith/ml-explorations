import torch
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
import logging

# logging.basicConfig(filename="log.log", level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

ITERATIONS = 10


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

    # TODO: implement side connections as shown in Fig3
    # TODO: will cloning weights mess up model?
    # TODO: implement forward pass with no training for initial timesteps
    # TODO: should_damp with false to init model layers will suffer from undamped changes in previously activated layers
    def forward_timestep(self, input_image, prev_activations, one_hot_labels, should_damp=True):
        for i in range(0, len(prev_activations)):
            prev_activations[i] = prev_activations[i].detach()

        # print("prev activations")
        # print(prev_activations)

        # prev_norm_act = prev_activations

        prev_norm_act = []
        for prev_act in prev_activations:
            if torch.all(prev_act == 0):
                prev_norm_act.append(prev_act)
                continue

            norm = prev_act.norm(p=2, dim=1, keepdim=True)
            prev_norm_act.append(prev_act.div(norm))

        # print("debug")
        # print(prev_norm_act)

        output_layer_weights = self.layers[-1].weight.t()
        new_activations = []
        if len(prev_activations) == 1:
            new_act = F.linear(
                one_hot_labels, output_layer_weights) + F.linear(input_image, self.layers[0].weight)
            new_act = (1 - self.damping_factor) * prev_activations[0] + \
                self.damping_factor * new_act
            new_activations.append(F.relu(new_act))
        else:
            for i, (prev_act, layer) in enumerate(zip(prev_activations, self.layers[:-1])):
                # first layer gets the input image
                if i == 0:
                    # print("------, ", i)
                    # print("first weights")
                    # print(input_image.shape)
                    # print(layer.weight.shape)

                    # print("second")
                    # print(prev_norm_act[i + 1].shape)
                    # print(self.layers[i+1].weight.t().shape)

                    new_activations.append(F.linear(input_image, layer.weight) + F.linear(
                        prev_norm_act[i + 1], self.layers[i+1].weight.t()))
                # last hidden layer gets the previous layer's activation and the one-hot labels
                elif i == len(prev_activations) - 1:
                    # print("------, ", i)
                    # print("first weights")
                    # print(prev_norm_act[i - 1].shape)
                    # print(layer.weight.shape)
                    # F.linear(prev_norm_act[i - 1], layer.weight)

                    # print("second")
                    # print(one_hot_labels.shape)
                    # print(output_layer_weights.shape)
                    # F.linear(one_hot_labels, output_layer_weights)

                    new_activations.append(F.linear(prev_norm_act[i - 1], layer.weight) + F.linear(
                        one_hot_labels, output_layer_weights))
                # other layers get activations from the layers above and below
                else:
                    new_activations.append(F.linear(prev_norm_act[i - 1], layer.weight) + F.linear(
                        prev_norm_act[i + 1], self.layers[i+1].weight.t()))

                if should_damp:
                    new_activations[i] = (1 - self.damping_factor) * prev_act + \
                        self.damping_factor * new_activations[i]

                new_activations[i] = F.relu(new_activations[i])

        # print("new activations")
        # print(new_activations)

        return new_activations


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


def train(model, positive_images, positive_labels, negative_images, negative_labels, optimizer, device, threshold=2):
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

    # with no grad
    with torch.no_grad():
        for iteration in range(0, len(model.layers[:-1])):
            positive_activations = model.forward_timestep(
                positive_images, positive_activations, positive_one_hot_labels, False)

            negative_activations = model.forward_timestep(
                negative_images, negative_activations, negative_one_hot_labels, False)

        # perform multiple iterations of the recurrent forward pass
    running_loss = 0
    for iteration in range(ITERATIONS):
        # print(torch.mps.current_allocated_memory() / (10 ** 9))

        # calculate positive goodness
        positive_activations = model.forward_timestep(
            positive_images, positive_activations, positive_one_hot_labels)
        # print("activations shape")
        # for act in positive_activations:
        #     print(act.shape)
        positive_goodness = activations_to_goodness(positive_activations)
        # print("goodness shape")
        # print(positive_goodness.shape)
        print("positive goodness")
        print(positive_goodness)

        # calculate negative goodness
        negative_activations = model.forward_timestep(
            negative_images, negative_activations, negative_one_hot_labels)
        negative_goodness = activations_to_goodness(negative_activations)
        print("negative goodness")
        print(negative_goodness)

        # train each layer independently
        layer_losses_len = len(model.layers[:-1])
        layer_losses_sum = 0
        for i, (pos_good, neg_good, _layer) in tqdm(enumerate(zip(positive_goodness, negative_goodness, model.layers[:-1]))):
            optimizer.zero_grad()
            layer_loss = 0

            # pos_good = pos_good.item()
            # neg_good = neg_good.item()
            # print("----------- ", pos_good.grad)

            # todo: consider simplifying loss function
            layer_loss = torch.log(1 + torch.exp(torch.cat([
                (-1 * pos_good) + threshold,
                neg_good - threshold
            ]))).mean()
            layer_losses_sum += layer_loss.item()

            layer_loss.backward()

            # TODO: something strange is going on with grad
            # tcount = 0
            # for name, param in model.named_parameters():
            #     if param.requires_grad:
            #         print(name, param.data)
            #         # print(name, param.grad)
            #         tcount += 1
            #     if tcount == 1:
            #         break
            # input()

            optimizer.step()
            optimizer.zero_grad()

            for i in range(0, len(positive_activations)):
                positive_activations[i] = positive_activations[i].detach()

            for i in range(0, len(negative_activations)):
                negative_activations[i] = negative_activations[i].detach()

        average_layer_loss = layer_losses_sum/layer_losses_len
        running_loss += average_layer_loss
        # print(torch.mps.current_allocated_memory() / (10 ** 9))
        logging.info(
            f'iteration {iteration+1}, average layer loss: {average_layer_loss}')

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
                    activations = model.forward_timestep(
                        images, activations, one_hot_labels, False)

                for _ in range(ITERATIONS):
                    activations = model.forward_timestep(
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
    device = torch.device("cpu")

    torch.autograd.set_detect_anomaly(True)

    torch.manual_seed(1234)

    train_loader, test_loader = MNIST_loaders()

    model = RecurrentFFNet(784, [500], 10).to(device)
    # TODO: decrease learning rate
    optimizer = Adam(model.parameters())
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
