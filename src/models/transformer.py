import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange

class CausalSelfAttention(nn.Module):
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
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q, k, v = map(
            lambda t: rearrange(t, "b t (h d) -> b h t d", h=self.n_head),
            (q, k, v)
        ) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)

        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = rearrange(y, "b h t d -> b t (h d)") # (B, T, C), re-assemble all head outputs side by side

        # output projection
        y = self.c_proj(y)
        return y

class TransformerBlock(nn.Module):
    def __init__(self, configs):
        super().__init__()

        self.n_embd = configs.transformer_dim

        self.ln_1 = nn.LayerNorm(self.n_embd)
        self.attn = CausalSelfAttention(configs)
        self.ln_2 = nn.LayerNorm(self.n_embd)

        self.mlp = nn.Sequential(
            nn.Linear(self.n_embd, 4 * self.n_embd),
            nn.GELU(),
            nn.Linear(4 * self.n_embd, self.n_embd)
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class MotionTransformer(nn.Module):
    def __init__(self, configs):
        super().__init__()

        self.configs = configs

        self.n_embd = configs.transformer_dim
        self.block_size = configs.input_length

        self.fc_in = nn.Linear(configs.motion_dim, self.n_embd)
        self.pos_emb = nn.Embedding(self.block_size, self.n_embd)

        self.transformer = nn.Sequential(*[
            TransformerBlock(configs)
            for _ in range(configs.transformer_num_layers)
        ])

        self.layer_norm = nn.LayerNorm(self.n_embd)
        self.fc_out = nn.Linear(self.n_embd, configs.motion_dim, bias=False)

    def forward(self, x):
        b, n, c = x.size()

        pos = torch.arange(n, device=x.device).unsqueeze(0) # (1, n)

        x = self.fc_in(x) + self.pos_emb(pos) # (b, n, embed_dim) + (1, n, embed_dim)

        x = self.transformer(x)

        x = self.layer_norm(x)
        x = self.fc_out(x)

        x = x[:, -self.configs.target_length_train:]

        return x
