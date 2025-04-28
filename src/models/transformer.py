import math
import torch
import torch.nn as nn
from torch.nn import functional as F

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, configs):
        super().__init__()

        assert configs.transformer_dim % configs.transformer_num_heads == 0

        self.n_head = configs.transformer_num_heads
        self.n_embd = configs.transformer_dim
        self.block_size = configs.input_length

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd)
        # output projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(self.block_size, self.block_size)).view(1, 1, self.block_size, self.block_size)
        )

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.c_proj(y)
        return y

class Block(nn.Module):
    def __init__(self, configs):
        super().__init__()

        self.n_embd = configs.transformer_dim

        self.ln_1 = nn.LayerNorm(self.n_embd)
        self.attn = CausalSelfAttention(configs)
        self.ln_2 = nn.LayerNorm(self.n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc   = nn.Linear(self.n_embd, 4 * self.n_embd),
            c_proj = nn.Linear(4 * self.n_embd, self.n_embd),
            act    = NewGELU()
        ))
        m = self.mlp
        self.mlpf = lambda x: m.c_proj(m.act(m.c_fc(x))) # MLP forward

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x

class MotionTransformer(nn.Module):
    def __init__(self, configs):
        super().__init__()

        self.configs = configs

        self.n_embd = configs.transformer_dim
        self.block_size = configs.input_length

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Linear(configs.motion_dim, self.n_embd),
            wpe = nn.Embedding(self.block_size, self.n_embd),
            h = nn.ModuleList([Block(configs) for _ in range(configs.transformer_num_layers)]),
            ln_f = nn.LayerNorm(self.n_embd)
        ))
        self.lm_head = nn.Linear(self.n_embd, configs.motion_dim, bias=False)

    def forward(self, x):
        b, n, c = x.size()

        pos = torch.arange(0, n, dtype=torch.long, device=x.device).unsqueeze(0) # (1, n)

        x_ = self.transformer.wte(x) # (b, n, embed_dim)
        pos_emb = self.transformer.wpe(pos) # (1, n, embed_dim)

        x_ = x_ + pos_emb

        for block in self.transformer.h:
            x_ = block(x_)

        x_ = self.transformer.ln_f(x_)
        x_ = self.lm_head(x_)

        x_ = x_[:, -self.configs.target_length_train:]

        return x_
