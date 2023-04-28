import torch
from torch import nn
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

ITERATIONS = 100 

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

def overlay_y_on_x(x, y):
    """Replace the first 10 pixels of data [x] with one-hot-encoded label [y]
    """
    x_ = x.clone()
    x_[:, :10] *= 0.0
    x_[range(x.shape[0]), y] = x.max()
    return x_


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

    # TODO: implement side connections as shown in Fig3
    def forward_timestep(self, input_image, prev_activations, one_hot_labels):
        new_activations = []
        output_layer_weights = self.layers[-1].weight.t()
        for i, (prev_act, layer, norm) in enumerate(zip(prev_activations, self.layers[:-1], self.layer_norms)):

            if i == 0:  # first layer gets the input image
                new_act = F.linear(input_image, layer.weight) + F.linear(prev_activations[i + 1], self.layers[i+1].weight.t())
            elif i == len(prev_activations) - 1:  # last hidden layer gets the previous layer's activation and the one-hot labels
                new_act = F.linear(prev_activations[i - 1], layer.weight) + F.linear(one_hot_labels, output_layer_weights)
            else:  # other layers get activations from the layers above and below
                new_act = F.linear(prev_activations[i - 1], layer.weight) + F.linear(prev_activations[i + 1], self.layers[i+1].weight.t())
            
            new_act = (1 - self.damping_factor) * prev_act + self.damping_factor * new_act
            new_act = norm(new_act)
            new_act = F.relu(new_act)
            new_activations.append(new_act.detach())

        return new_activations
    

def activations_to_goodness(activations):
    return torch.tensor([torch.mean(torch.square(act)) for act in activations]).requires_grad_()

# TODO: rename images to inputs
# TODO: consider adding a way to ignore layer training if activations haven't reached during start of timestep processing
def train(model, positive_images, positive_labels, negative_images, negative_labels, optimizer, device, threshold=2):
    model.train()

    # Prepare positive one-hot encoded labels
    positive_one_hot_labels = torch.zeros(len(positive_labels), model.num_classes, device=device)
    positive_one_hot_labels.scatter_(1, positive_labels.unsqueeze(1), 1.0)

    # Prepare negative one-hot encoded labels
    negative_one_hot_labels = torch.zeros(len(negative_labels), model.num_classes, device=device)
    negative_one_hot_labels.scatter_(1, negative_labels.unsqueeze(1), 1.0)

    # Initialize activations with zeros
    # TODO: dedup
    positive_activations = [torch.zeros(positive_images.shape[0], layer.out_features, device=device) for layer in model.layers[:-1]]
    negative_activations = [torch.zeros(positive_images.shape[0], layer.out_features, device=device) for layer in model.layers[:-1]]

    # Perform multiple iterations of the recurrent forward pass
    running_loss = 0
    for iteration in range(ITERATIONS):
        print(torch.mps.current_allocated_memory())
        optimizer.zero_grad()

        # Calculate positive goodness
        positive_activations = model.forward_timestep(positive_images, positive_activations, positive_one_hot_labels)
        positive_goodness = activations_to_goodness(positive_activations)
        # print("***************************************")
        # print("positive goodness")
        # print(positive_goodness)

        # Calculate negative goodness
        negative_activations = model.forward_timestep(negative_images, negative_activations, negative_one_hot_labels)
        negative_goodness = activations_to_goodness(negative_activations)
        # print("negative goodness")
        # print(negative_goodness)

        # Train each layer independently
        layer_losses = []
        for i, (pos_good, neg_good, layer) in tqdm(enumerate(zip(positive_goodness, negative_goodness, model.layers[:-1]))):
            optimizer.zero_grad()
            layer_loss = 0

            pos_good = pos_good.item()
            neg_good = neg_good.item()

            # TODO: consider simplifying loss function
            # TODO: consider using loss function from pytorch upstreamed
            layer_loss = torch.log(1 + torch.exp(torch.cat([
                -1 * positive_goodness + threshold, 
                negative_goodness - threshold]))).mean()
            layer_losses.append(layer_loss.item())

            # TODO: confirm we don't need retain_graph=True
            layer_loss.backward()
            optimizer.step()

        average_layer_loss = sum(layer_losses)/len(layer_losses) 
        running_loss += average_layer_loss
        logging.info(f'Iteration {iteration+1}, Average Layer Loss: {average_layer_loss}')
        # print("-----------------------------------")


    return running_loss / ITERATIONS

def test(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            all_labels_goodness = []
            
            # Evaluate goodness for each possible label
            for label in range(model.num_classes):
                one_hot_labels = torch.zeros(images.shape[0], model.num_classes, device=device)
                one_hot_labels[:, label] = 1.0
                activations = [torch.zeros(images.shape[0], layer.out_features, device=device) for layer in model.layers[:-1]]

                for _ in range(ITERATIONS):
                    activations = model.forward_timestep(images, activations, one_hot_labels)

                goodness = activations_to_goodness(activations).mean()
                all_labels_goodness.append(goodness)

            # Select the label with the maximum goodness
            predicted_labels = torch.argmax(torch.stack(all_labels_goodness), dim=0)
            
            total += images.size(0)
            correct += (predicted_labels == labels).sum().item()

    accuracy = 100 * correct / total
    logging.info(f'Test Accuracy: {accuracy}%')

    return accuracy

if __name__ == "__main__":
    # this ensures that the current MacOS version is at least 12.3+
    assert(torch.backends.mps.is_available())
    # this ensures that the current current PyTorch installation was built with MPS activated.
    assert(torch.backends.mps.is_built())
    # set the output to device: mps
    device = torch.device("mps")

    torch.manual_seed(1234)
    train_loader, test_loader = MNIST_loaders()

    model = RecurrentFFNet(784, [500, 250, 100], 10).to(device)
    optimizer = Adam(model.parameters())
    num_epochs = 3 

    for epoch in range(num_epochs):
        logging.info(f'Training starting...')

        x, y = next(iter(train_loader))
        x, y = x.to(device), y.to(device)

        x_pos = overlay_y_on_x(x, y)
        y_pos = y

        shuffled_labels = torch.randperm(x.size(0))
        y_neg = y[shuffled_labels]
        x_neg = overlay_y_on_x(x, y_neg)

        logging.info(f'x shape: {x_pos.shape}')

        loss = train(model, x_pos, y_pos, x_neg, y_neg, optimizer, device)
        logging.info(f'Epoch {epoch+1}, Loss: {loss}')

        accuracy = test(model, test_loader, device)
        logging.info(f'Epoch {epoch+1}, Loss: {loss}, Accuracy: {accuracy}')
