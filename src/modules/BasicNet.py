import torch.nn as nn
import torch.nn.functional as F

class BasicNet(nn.Module):
    def __init__(self, in_size, out_size, num_layers=4, layer_size=32, dropout_rate=0.5):
        super(BasicNet, self).__init__()

        self.layers = nn.ModuleList(
            [nn.Linear(in_size, layer_size)] +
            [nn.Linear(layer_size, layer_size) for i in range(num_layers-1)] +
            [nn.Linear(layer_size, out_size)]
        )

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.dropout(x)
            x = F.relu(layer(x))
        x = self.dropout(x)
        x = self.layers[-1](x)
        return x
