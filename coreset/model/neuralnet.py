import torch
from torch import nn

# import torchtext


class MLP(nn.Module):
    n_neurons = 32

    def __init__(self, input_size, vocab_size=0, n_layers=3):
        super().__init__()
        self.shape = [self.n_neurons, self.n_neurons]
        self.activation = nn.ReLU()
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, 64),
            self.activation,
        )
        self.hidden = nn.Sequential(
            nn.Linear(64, 64),
            self.activation,
            nn.Linear(64, 64),
            self.activation,
            nn.Linear(64, 64),
            self.activation,
            nn.Linear(64, 64),
            self.activation,
            nn.Linear(64, 64),
            self.activation,
            nn.Linear(64, 32),
            self.activation,
            nn.Linear(32, self.n_neurons),
            self.activation,
        )

        self.preput = nn.Sequential(
            nn.Linear(self.n_neurons, 1),
            nn.Sigmoid(),
        )

        # self.output = nn.Sigmoid()

    def forward(self, features):
        x = self.input_layer(features)
        x = self.hidden(x)
        x = self.preput(x)
        return x
