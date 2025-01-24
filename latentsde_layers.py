import torch
import torch.nn as nn
import math
from collections import OrderedDict
import Dev.torchsde_addfloat as torchsde
from torchdiffeq import odeint_adjoint as odeint
    

class MLP(nn.Module):
    def __init__(self, input_dim, dims):
        super(MLP, self).__init__()

        self.net_ = []
        for i in range(len(dims)): 
            if i == 0: 
                self.net_.append(('linear{}'.format(i+1), nn.Linear(input_dim, dims[i])))
            else: 
                self.net_.append(('linear{}'.format(i+1), nn.Linear(dims[i-1], dims[i]))) 
            self.net_.append(('{}{}'.format('relu', i+1), nn.LeakyReLU()))
        self.net_.append(('linear', nn.Linear(dims[-1], input_dim, bias = True)))
        self.net_ = OrderedDict(self.net_)
        self.net = nn.Sequential(self.net_)

    def forward(self, x):
        x = self.net(x)  
        return x
    


class Drift_intra(torch.nn.Module):
    def __init__(self, args):
        super(Drift_intra, self).__init__()

        self.net = MLP(args.latent_dim, args.intra_dims)

    def forward(self, t, x):
        f_intra = self.net(x)  
        return f_intra
    

    
class SelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, attn_drop_ratio=0):

        super(SelfAttention, self).__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(attn_drop_ratio)

    def forward(self, x1, x2, x3):

        batch_size, d_model = x1.size()

        q = self.q_proj(x1).view(batch_size, self.n_heads, self.d_k).permute(1, 0, 2)  
        k = self.k_proj(x2).view(batch_size, self.n_heads, self.d_k).permute(1, 0, 2)  
        v = self.v_proj(x3).view(batch_size, self.n_heads, self.d_k).permute(1, 0, 2)  

        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)  
        mask = torch.eye(attn.size(-1), device=attn.device).bool()  
        mask = mask.unsqueeze(0).expand(attn.size(0), -1, -1)  
        attn = attn.masked_fill(mask, float('-inf'))
        
        attn = torch.softmax(attn, dim=-1)  
        attn = self.attn_drop(attn)

        output = torch.matmul(attn, v)  
        output = output.permute(1, 0, 2).reshape(batch_size, d_model) 

        return output, attn, q, k, v
    


class Block(nn.Module):
    def __init__(self,
                 d_model, 
                 num_heads, 
                 attn_drop_ratio=0.):
        super(Block, self).__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = SelfAttention(d_model, num_heads, attn_drop_ratio=attn_drop_ratio)


        self.Vnet = nn.LayerNorm(d_model)

        self.use_FFN = use_FFN
        self.norm2 = nn.LayerNorm(d_model)
        mlp_hidden_dim = int(d_model * 4)
        self.fc1 = nn.Linear(d_model, mlp_hidden_dim)
        self.act = nn.LeakyReLU()
        self.fc2 = nn.Linear(mlp_hidden_dim, d_model)

    def forward(self, x):
        v_initial = self.Vnet(x)
        x, weights, q, k, v = self.attn(self.norm1(x),self.norm1(x), v_initial)

        x = self.norm2(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)

        return x, weights, q, k, v
  


class Drift_inter(torch.nn.Module):
    def __init__(self, args):
        super(Drift_inter, self).__init__()
 
        self.trm_inter = nn.ModuleList(
                            [Block(d_model=args.latent_dim, num_heads=args.inter_trm_num_head, 
                                  attn_drop_ratio=args.attn_drop_ratio) 
                            for _ in range(args.inter_trm_num_layers)]
                            )

    def forward(self, t, x):

        for layer in self.trm_inter:
            x = layer(x)[0]

        return x









class LipSwish(nn.Module):
    def forward(self, x):
        return 0.909 * nn.functional.silu(x)

class Mlp(nn.Module):
    def __init__(self, input_dim, out_dim, hidden_dim, num_layers, tanh):
        super().__init__()

        model = [nn.Linear(input_dim, hidden_dim), LipSwish()]
        for _ in range(num_layers - 1):
            model.append(nn.Linear(hidden_dim, hidden_dim))
            model.append(LipSwish())
        model.append(nn.Linear(hidden_dim, out_dim))
        if tanh:
            model.append(nn.Tanh())
        self._model = nn.Sequential(*model)

    def forward(self, x):
        return self._model(x)
    





class Diffusion(torch.nn.Module):
    def __init__(self, args):
        super(Diffusion, self).__init__()

        self.dim = args.latent_dim
        self.sigma_const = args.sigma_const
        self.sigma_type = args.sigma_type

        cfg = dict(
            input_dim = self.dim,
            out_dim = self.dim,
            hidden_dim = 256,
            num_layers = 2,
            tanh=True
        )


        if self.sigma_type == 'Mlp':
            self.sigma = Mlp(**cfg)
        elif self.sigma_type == "const":
            self.register_buffer('sigma', torch.as_tensor(self.sigma_const))
            self.sigma = self.sigma.repeat(self.dim).unsqueeze(0)
        elif self.sigma_type == "const_param":
            self.sigma = nn.Parameter(torch.randn(1,self.dim), requires_grad=True)
 
    def forward(self, t, x):
        if self.sigma_type == "diagonal":
            g = self.sigma(x).view(-1, self.dim)
        elif self.sigma_type == "const":
            g = self.sigma.repeat(x.shape[0], 1)
        elif self.sigma_type == "const-param":
            g = self.sigma.repeat(x.shape[0], 1)
        return g


