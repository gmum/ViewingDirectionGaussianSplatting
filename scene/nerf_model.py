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
        self.main = tcnn.Network(input_size + self.enc.n_output_dims, output_size, network_config=config["vdgs"]) 

        def init_weights(m):
            # if isinstance(m, nn.Linear):
            torch.nn.init.ones_(m.params.data)
                # m.bias.data.fill_(1.0)
                
        # self.enc.apply(init_weights)
        self.main.apply(init_weights)

    def forward(self, shs, rotations, scales, viewdirs, xyz):
        shs = shs.view(shs.size(0), -1)
        shs = torch.nn.functional.normalize(shs)
        
        emb = self.enc(xyz)
        x = torch.concat([shs, viewdirs, rotations, scales, emb], dim=1)
        x = self.main(x)

        return x
