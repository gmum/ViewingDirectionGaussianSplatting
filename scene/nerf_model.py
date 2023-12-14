import torch
import torch.nn as nn


class Embedder:
    def __init__(self):
        self.include_input = False
        self.input_dims = 3
        self.max_freq_log2 = 3
        self.num_freqs = 4
        self.log_sampling = True
        self.periodic_fns = [torch.sin, torch.cos]
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        out_dim = 0
        if self.include_input:
            embed_fns.append(lambda x : x)
            out_dim += self.input_dims
            
        max_freq = self.max_freq_log2
        N_freqs = self.num_freqs = 4
        
        if self.log_sampling:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.periodic_fns:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += self.input_dims
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


class MLP(nn.Module):
    def __init__(self, input_size, output_size, target = ""):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.target = target
        self.slope = 0.01
        W = 32
        self.main = nn.Sequential(
                nn.Linear(self.input_size, W*2),
                nn.LeakyReLU(self.slope),
                nn.Linear(W*2, W),
                nn.LeakyReLU(self.slope),
                nn.Linear(W, W),
            )
        self.rotation = nn.Sequential(nn.Linear(W, self.output_size), nn.Sigmoid())
        self.out = nn.Sequential(nn.Linear(W, 3), nn.Sigmoid())
        self.alpha = nn.Sequential(nn.Linear(W, 1), nn.Sigmoid())

    def forward(self, x, rotations, scales, y):
        x = x.view(x.size(0), -1) 
        x = torch.nn.functional.normalize(x)
        x = torch.concat([x, y, rotations, scales], dim=1)
        x = self.main(x)
        #if self.target == "rotation":
        #    return self.rotation(x)
        return self.out(x)