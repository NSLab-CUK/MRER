import torch
import torch.nn as nn
from torch.nn import functional as F


class TemporalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        self.c_attn = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.temp = nn.Parameter(torch.ones(config.n_head,1))
        self.denom_bias = nn.Parameter(torch.zeros(config.n_head, config.block_size,1))
    
    def forward(self, x):
        B, T, C = x.size() 

        w = self.c_attn(x)
        w = w.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) 
        w_sq = w ** 2
        denom = (torch.cumsum(w_sq,dim=-2)).clamp_min(1e-12)
        w_normed = (w_sq / denom) + self.denom_bias[:,:T,:]
        tmp = torch.sum(w_normed, dim=-1)* self.temp
        
        Pi = F.softmax(tmp, dim=1) 
        dots = torch.cumsum(w_sq * Pi.unsqueeze(-1), dim=-2) / (Pi.cumsum(dim=-1) + 1e-8).unsqueeze(-1)
        attn = 1. / (1 + dots)       
        attn = self.attn_dropout(attn)
        y = - torch.mul(w.mul(Pi.unsqueeze(-1)), attn)
        y = y.transpose(1, 2).contiguous().view(B, T, C) 
        y = self.resid_dropout(self.c_proj(y))
        return y
    

class AttentionConfig:

    def __init__(self, n_embd, n_head, bias, dropout, block_size):
        self.n_embd = n_embd
        self.n_head = n_head
        self.bias = bias
        self.dropout = dropout
        self.block_size = block_size
