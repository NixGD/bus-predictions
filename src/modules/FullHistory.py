import torch
import torch.nn as nn

from .BasicNet import BasicNet


class FullHistory(nn.Module):
    def __init__(self, encode_hist=True, encoding_size=3, num_layers=4, layer_size=32, dropout_rate=0.5):
        super(FullHistory, self).__init__()

        self.hist_input_size = 5
        self.trip_input_size = 9
        self.encode_hist = encode_hist
        self.encoding_size = encoding_size

        if self.encode_hist:
            self.encoder = BasicNet(
                in_size=self.hist_input_size,
                out_size=self.encoding_size,
                num_layers=3,
                layer_size=layer_size,
                dropout_rate=dropout_rate)

        hist_out_size = self.encoding_size if self.encode_hist else self.hist_input_size
        self.prediction_net = BasicNet(
            in_size=self.trip_input_size + hist_out_size,
            out_size=1,
            num_layers=num_layers,
            layer_size=layer_size,
            dropout_rate=dropout_rate
        )

    def forward(self, trip, hist):
        if self.encode_hist:
            hist = self.encoder(hist)
        x = torch.cat([trip, hist], axis=1)
        x = self.prediction_net(x)
        return x
