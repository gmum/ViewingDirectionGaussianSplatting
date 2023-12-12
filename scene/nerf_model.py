import torch

class VaniliaNeRF(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.xyz_layer = torch.nn.Sequential(
            torch.nn.Linear(3, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 48),
            torch.nn.ReLU()
        )


    def forward(self, shs, xyz):
        xyz = self.xyz_layer(xyz)
        xyz = xyz.view(-1, 16, 3)
        return shs * xyz
