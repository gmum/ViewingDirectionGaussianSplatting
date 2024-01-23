import torch
import torch.nn as nn


class VDGS(nn.Module):
    def __init__(self, input_size: int, output_size: int = 1, net_width: int = 32):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.net_width = net_width
        self.factor = 2

        self.main = nn.Sequential(
            nn.Linear(self.input_size, self.net_width * self.factor),
            nn.LeakyReLU(),
            nn.Linear(self.net_width * self.factor, self.net_width),
            nn.LeakyReLU(),
            nn.Linear(self.net_width, self.net_width),
        )
        self.head = nn.Sequential(
            nn.Linear(self.net_width, self.output_size), nn.Tanh()
        )

    def forward(self, shs, rotations, scales, viewdirs):
        shs = shs.view(shs.size(0), -1)
        shs = torch.nn.functional.normalize(shs)
        x = torch.concat([shs, viewdirs, rotations, scales], dim=1)
        x = self.main(x)
        x =  self.head(x)
        return x
