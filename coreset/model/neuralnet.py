import torch
from torch import nn

# import torchtext


class MLP(nn.Module):
    n_neurons = 32

    def __init__(self, input_size, n_layers=3):
        super().__init__()
        self.shape = [self.n_neurons, self.n_neurons]
        self.activation = nn.ReLU()
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, self.n_neurons),
            # self.activation,
        )
        self.hidden = nn.Sequential(
            nn.Linear(32, 64),
            self.activation,
            nn.Linear(64, 128),
            self.activation,
            nn.Linear(128, 128),
            self.activation,
            nn.Linear(128, 64),
            self.activation,
            nn.Linear(64, 64),
            self.activation,
            nn.Linear(64, 32),
            # self.activation,
        )
        # self.hidden = nn.Bilinear(self.n_neurons, self.n_neurons, self.n_neurons)

        self.preput = nn.Sequential(
            # nn.Linear(self.n_neurons, 1),
            # self.activation,
            nn.Linear(self.n_neurons, 10),
            # self.activation,
            nn.Softmax(dim=1),
        )
        # self.output = nn.Softmax(dim=1)

        # self.output = nn.Sigmoid()

    def forward(self, features):
        x = self.input_layer(features)
        # x = self.hidden(x, x)
        x = self.hidden(x)
        x = self.preput(x)
        # x = self.output(x)
        # x = torch.argmax(x, dim=1).float().reshape(-1, 1)
        return x
        # return torch.sigmoid(x)
        # return torch.softmax(x, dim=1)
        # return torch.log_softmax(x, dim=1)
