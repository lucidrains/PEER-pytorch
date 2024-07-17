from math import sqrt

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList

import einx
from einops import einsum, pack, unpack
from einops.layers.torch import Rearrange

from PEER_pytorch.PK import PK

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

# rmsnorm

class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.scale * (self.gamma + 1)

# main class

class PKAttention(Module):
    def __init__(
        self,
        dim,
        *,
        causal = True,
        heads = 8,
        num_key_values = 1_000_000,
        key_value_pk_topk = 16,
        dim_key = None,
        product_keys = 2,
        pre_rmsnorm = False,
        dropout = 0.
    ):
        """
        einops notation
        b - batch
        n - sequence
        d - dimension
        h - heads
        p - 2 for product key
        k - number of keys
        """

        super().__init__()
        self.causal = causal
        self.heads = heads
        self.num_key_values = num_key_values

        self.norm = RMSNorm(dim) if pre_rmsnorm else nn.Identity()

        # experts that will form the mlp project in / out weights

        self.to_queries = nn.Sequential(
            nn.Linear(dim, dim * heads, bias = False),
            Rearrange('b n (h d) -> b n h d', h = heads)
        )

        # keys and values selected using product-key

        self.keys = nn.EmbeddingBag(num_key_values * heads, dim, mode = 'sum')
        self.values = nn.EmbeddingBag(num_key_values * heads, dim, mode = 'sum')

        assert sqrt(num_key_values).is_integer(), '`num_key_values` needs to be a square'
        assert (dim % 2) == 0, 'feature dimension should be divisible by 2'

        self.to_kv_pk_indices = PK(
            dim = dim,
            num_keys = int(sqrt(num_key_values)),
            final_topk = key_value_pk_topk,
            product_keys = product_keys
        )

        # dropout

        self.dropout = nn.Dropout(dropout)

        # output

        self.to_out = nn.Sequential(
            Rearrange('b h n d -> b n (h d)'),
            nn.Linear(dim * heads, dim, bias = False)
        )

    def forward(
        self,
        x,
        mask = None
    ):
        device = x.device

        x = self.norm(x)

        # queries

        q = self.to_queries(x)

        q = q * (q.shape[-1] ** -0.5)

        # keys and values

        kv_scores, indices = self.to_kv_pk_indices(x, softmax_scores = True)

        offsets = torch.arange(self.heads, device = device) * self.num_key_values
        indices = einx.add('b n h k, h -> b n h k', indices, offsets)

        indices, packed_shape = pack_one(indices, '* k')
        kv_scores, _ = pack_one(kv_scores, '* k')

        k, v = self.keys(indices, per_sample_weights = kv_scores), self.values(indices, per_sample_weights = kv_scores)

        k = unpack_one(k, packed_shape, '* d')
        v = unpack_one(v, packed_shape, '* d')

        # usual multihead self attention

        sim = einsum(q, k, 'b i h d, b j h d -> b h i j')

        # whether causal or not

        if self.causal:
            assert not exists(mask)
            i, j, device = *sim.shape[-2:], x.device
            causal_mask = torch.ones((i, j), device = device, dtype = torch.bool).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        elif exists(mask):
            sim = einx.where('b j, b h i j, -> b h i j', mask, sim, -torch.finfo(sim.dtype).max)

        # attention

        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        # aggregate

        out = einsum(attn, v, 'b h i j, b j h d -> b h i d')

        # combine heads

        return self.to_out(out)

# main

if __name__ == '__main__':
    peer_attn = PKAttention(
        dim = 256,
        causal = True,
        heads = 8,
        num_key_values = int(1e4),
        pre_rmsnorm = True
    )

    x = torch.randn(2, 512, 256)

    out = peer_attn(x) + x

    assert x.shape == out.shape
