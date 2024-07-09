from math import sqrt

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList

import einx
from einops import einsum
from einops.layers.torch import Rearrange

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# main class

class PEER(Module):
    """
    following Algorithm 1 in the paper
    """

    def __init__(
        self,
        dim,
        *,
        heads = 8,                   # tested up to 32 - (hk = heads * num_experts_per_head (16))
        num_experts = 1_000_000,     # he chose 1 million
        num_experts_per_head = 16,   # he settled on 16, but was 32 in PKM paper
        activation = nn.GELU,
        dim_key = 128,
        product_key_topk = None
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

        # experts that will form the mlp project in / out weights

        self.weight_down_embed = nn.Embedding(num_experts, dim)
        self.weight_up_embed = nn.Embedding(num_experts, dim)

        # activation function, defaults to gelu

        self.activation = activation()

        # queries and keys for product-key

        assert sqrt(num_experts).is_integer(), '`num_experts` needs to be a square'

        self.num_keys = int(sqrt(num_experts))

        self.to_queries = nn.Sequential(
            nn.Linear(dim, dim_key * heads * 2, bias = False),
            Rearrange('b n (p h d) -> p b n h d', p = 2, h = heads)
        )

        self.product_key_topk = default(product_key_topk, num_experts_per_head)
        self.num_experts_per_head = num_experts_per_head

        self.keys = nn.Parameter(torch.zeros(heads, self.num_keys, 2, dim_key))
        nn.init.normal_(self.keys, std = 0.02)

    def forward(
        self,
        x
    ):
        # queries

        queries = self.to_queries(x)

        # first get similarity with keys

        sim = einsum(queries, self.keys, 'p b n h d, h k p d -> p b n h k')

        # product key logic

        (scores_x, scores_y), (indices_x, indices_y) = sim.topk(self.product_key_topk, dim = -1)

        all_scores = einx.add('... i, ... j -> ... (i j)', scores_x, scores_y)
        all_indices = einx.add('... i, ... j -> ... (i j)', indices_x * self.num_keys, indices_y)

        scores, indices = all_scores.topk(self.num_experts_per_head, dim = -1)

        # build the weight matrices for projecting in and out
        # basically the experts are the gathered parameters for an MLP

        weights_down = self.weight_down_embed(indices)
        weights_up = self.weight_up_embed(indices)

        # below is basically Algorithm 1 in paper

        x = einsum(x, weights_down, 'b n d, b n h k d -> b n h k')

        x = self.activation(x)

        x = x * scores.softmax(dim = -1)

        x = einsum(x, weights_up, 'b n h k, b n h k d -> b n d')

        return x
