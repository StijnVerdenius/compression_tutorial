from typing import Any

import torch.nn as nn

from PrunableModel import PrunableModel


class FFNET(PrunableModel):

    def __init__(self, **kwargs):
        super(FFNET, self).__init__(**kwargs)

        self.layers = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 2)
        )

        self._post_init()

    def forward(self, x):
        return self.layers.forward(x)



