import torch
import torch.nn as nn


class VDGS(nn.Module):
    def __init__(self, input_size: int, output_size: int = 1, net_width: int = 32):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.net_width = net_width
        self.factor = 2

        import commentjson as json
        import tinycudann as tcnn

        with open("config.json") as f:
            config = json.load(f)
        
        self.enc = tcnn.Encoding(3, encoding_config=config["encoding"])
        self.main = tcnn.Network( self.enc.n_output_dims + 3, output_size, network_config=config["vdgs"]) 

    def forward(self, viewdirs, xyz):
        emb = self.enc(xyz)
        x = torch.concat([viewdirs, emb], dim=1)
        x = self.main(x)

        return x
