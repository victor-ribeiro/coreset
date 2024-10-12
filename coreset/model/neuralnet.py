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
            nn.Linear(input_size, 512),
            self.activation,
            # nn.EmbeddingBag(vocab_size, embedding_dim=256),
            # nn.LayerNorm(256, 256),
        )
        self.hidden = nn.Sequential(
            nn.Linear(512, 512),
            self.activation,
            nn.Linear(512, 512),
            self.activation,
            nn.Linear(512, 128),
            self.activation,
            nn.Linear(128, 128),
            self.activation,
            nn.Linear(128, 64),
            self.activation,
            nn.Linear(64, self.n_neurons),
            self.activation,
        )
        # self.hidden = nn.Bilinear(self.n_neurons, self.n_neurons, self.n_neurons)

        self.preput = nn.Sequential(
            # nn.Linear(self.n_neurons, 2),
            nn.Linear(self.n_neurons, 1),
            # nn.Softmax(dim=1),
            nn.Sigmoid(),
        )

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
