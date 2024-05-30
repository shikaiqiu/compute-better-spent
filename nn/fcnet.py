import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .cola_nn import dense_init


class CappedList():
    # used for caching activations for logging
    def __init__(self, max_len=1):
        self.max_len = max_len
        self.buffer = []

    def append(self, x):
        if len(self.buffer) < self.max_len:
            self.buffer.append(x.cpu())

class MLPBlock(nn.Module):
    def __init__(self, width, residual, layer_norm, residual_mult=1, use_bias=True):
        super().__init__()
        self.linear1 = nn.Linear(width, width * 4, bias=use_bias)
        self.linear2 = nn.Linear(width * 4, width, bias=use_bias)
        self.residual_mult = residual_mult
        dense_init(self.linear1)
        dense_init(self.linear2, zero_init=True)
        self.residual = residual
        self.layer_norm = layer_norm
        if layer_norm:
            self.ln = nn.LayerNorm(width)
        self.gelu = nn.GELU()

    def forward(self, x):
        x0 = x
        if self.layer_norm:
            x = self.ln(x)
        x = self.gelu(self.linear1(x))
        x = self.linear2(x) * self.residual_mult
        if self.residual:
            x = x + x0
        return x


class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, depth, width, residual=True, layer_norm=True, shuffle_pixels=True, attn_mult=1,
                 output_mult=1, emb_mult=1, use_bias=True, **_):
        super().__init__()
        self.shuffle_pixels = shuffle_pixels
        local_rng = np.random.default_rng(42)
        # Shuffle the pixels using the local random generator
        self.pixel_indices = local_rng.permutation(dim_in)
        self.output_mult = output_mult
        self.emb_mult = emb_mult
        # input layer
        self.input_layer = nn.Linear(dim_in, width, bias=use_bias)
        dense_init(self.input_layer)
        # hidden layers
        self.hidden_layers = nn.ModuleList()
        for _ in range(depth):
            self.hidden_layers.append(MLPBlock(width, residual, layer_norm, residual_mult=attn_mult, use_bias=use_bias))
        # output layer
        self.output_layer = nn.Linear(width, dim_out, bias=use_bias)
        dense_init(self.output_layer, zero_init=True)
        # logs
        self.hs = [CappedList() for _ in range(depth + 2)]

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        if self.shuffle_pixels:
            x = x[:, self.pixel_indices]
        x = F.gelu(self.input_layer(x) * self.emb_mult)
        if not self.training:
            self.hs[0].append(x.detach())
        for i, layer in enumerate(self.hidden_layers):
            x = layer(x)
            if not self.training:
                self.hs[i + 1].append(x.detach())
        y = self.output_layer(x) * self.output_mult
        if not self.training:
            self.hs[-1].append(y.detach())
        return y

    def get_features(self):
        return self.hs

    def clear_features(self):
        self.hs = [CappedList() for _ in range(len(self.hs))]