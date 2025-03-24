from __future__ import annotations

from torch import nn


class MLP(nn.Module):
    def __init__(self, sizes):
        super().__init__()
        layers = []
        for i, (in_size, out_size) in enumerate(zip(sizes[:-1], sizes[1:])):
            layers.append(nn.Linear(in_size, out_size))
            if i < len(sizes) - 2:
                layers.append(nn.LeakyReLU(0.1))
                layers.append(nn.Dropout(0.1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
