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
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.slope = 0.01

        self.main = nn.Sequential(
                nn.Linear(self.input_size, 64),
                nn.LeakyReLU(self.slope),
                nn.Linear(64, 64),
                nn.LeakyReLU(self.slope),
            )
        self.opactiy = nn.Sequential(
            nn.Linear(4, 64),
            nn.LeakyReLU(self.slope),
            nn.Linear(64, 64),
            nn.LeakyReLU(self.slope),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self.shs_factor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(48, 48),
            nn.LeakyReLU(self.slope),
            nn.Linear(48, 48),
            nn.Sigmoid()
        )

    def forward(self, opacity, viewdir):
        # shs = shs.view(shs.size(0), -1) # 48 + 54 = 102
        x = torch.concat([opacity, viewdir], dim=1)
        # x = self.main(x)
        opacity = self.opactiy(x)
        # shs_new = self.shs_factor(shs)
        # shs_new = shs_new.view(-1, 16, 3)
        return opacity
    
    def get_parameters(self):
        param_list = []
        for _, param in self.named_parameters():
            param_list.append(param)
        return param_list
    
    def pos(self, x):
        raise NotImplementedError
