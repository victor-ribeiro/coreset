import torch
from torch import nn

# import torchtext


class MLP(nn.Module):
    n_neurons = 32

    def __init__(self, input_size, output_size=1):
        super().__init__()
        self.shape = [self.n_neurons, self.n_neurons]
        self.activation = nn.Sigmoid()
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, 100),
            self.activation,
        )
        self.hidden = nn.Sequential(nn.Linear(100, 10), self.activation)

        self.preput = nn.Sequential(
            nn.Linear(10, output_size),
            nn.Sigmoid(),
        )

        # self.output = nn.Sigmoid()

    def forward(self, features):
        x = self.input_layer(features)
        x = self.hidden(x)
        x = self.preput(x)
        return x
