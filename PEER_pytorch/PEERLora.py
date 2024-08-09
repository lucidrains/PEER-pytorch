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

# rmsnorm

class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.scale * (self.gamma + 1)

# main class

class PEERLora(Module):
    """
    Same as PEER, except it retrieves LORA weights and adds them to a usual feedforward weight1 and weight2 matrices
    """

    def __init__(
        self,
        dim,
        *,
        expansion_factor = 2.,
        num_experts = 1_000_000,         # 1 million experts
        heads = 4,                       # the lora k dimension is kept at 16 (heads [4] * num_experts_per_head [4])
        num_experts_per_head = 4,
        activation = nn.GELU,
        dim_key = None,
        product_key_topk = None,
        pre_rmsnorm = False,
        non_competing_scores = True,
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
        dim_inner = int(dim * expansion_factor)

        self.norm = RMSNorm(dim) if pre_rmsnorm else nn.Identity()

        # heads and num experts

        self.heads = heads
        self.num_experts_per_head = num_experts_per_head

        self.num_experts = num_experts

        # usual feedforward weights without bias

        self.proj_in = nn.Linear(dim, dim_inner, bias = False)
        self.proj_out = nn.Linear(dim_inner, dim, bias = False)

        # experts that will form the mlp project in / out weights

        self.proj_in_lora_a = nn.Embedding(num_experts, dim)
        self.proj_in_lora_b = nn.Embedding(num_experts, dim_inner)

        self.proj_out_lora_a = nn.Embedding(num_experts, dim_inner)
        self.proj_out_lora_b = nn.Embedding(num_experts, dim)

        # activation function, defaults to gelu

        self.activation = activation()

        # queries and keys for product-key

        assert sqrt(num_experts).is_integer(), '`num_experts` needs to be a square'
        assert (dim % 2) == 0, 'feature dimension should be divisible by 2'

        dim_key = default(dim_key, dim // 2)
        self.num_keys = int(sqrt(num_experts))

        self.to_queries = nn.Sequential(
            nn.Linear(dim, dim_key * heads * 2, bias = False),
            Rearrange('b n (p h d) -> p b n h d', p = 2, h = heads)
        )

        self.product_key_topk = default(product_key_topk, num_experts_per_head)
        self.num_experts_per_head_topk = num_experts_per_head if not non_competing_scores else 1

        self.keys = nn.Parameter(torch.zeros(heads, self.num_keys, 2, dim_key))
        nn.init.normal_(self.keys, std = 0.02)

        # dropout

        self.dropout = nn.Dropout(dropout)

        # whether to use softmax on scores

        # Csordas et al claims non-competing activation helps in PKM setting
        # https://arxiv.org/pdf/2310.10837 - Table 2 in Section 6.2

        self.score_activation = nn.Softmax(dim = -1) if not non_competing_scores else nn.ReLU()

    @property
    def lora_k(self):
        return self.heads * self.num_experts_per_head

    def forward(
        self,
        x
    ):

        x = self.norm(x)

        # queries

        queries = self.to_queries(x)

        # first get similarity with keys

        sim = einsum(queries, self.keys, 'p b n h d, h k p d -> p b n h k')

        # product key logic

        (scores_x, scores_y), (indices_x, indices_y) = sim.topk(self.product_key_topk, dim = -1)

        all_scores = einx.add('... i, ... j -> ... (i j)', scores_x, scores_y)
        all_indices = einx.add('... i, ... j -> ... (i j)', indices_x * self.num_keys, indices_y)

        scores, pk_indices = all_scores.topk(self.num_experts_per_head_topk, dim = -1)

        indices = all_indices.gather(-1, pk_indices)

        # build the loras for projecting in and out weights of a feedforward

        proj_in_lora_a = self.proj_in_lora_a(indices)
        proj_in_lora_b = self.proj_in_lora_b(indices)

        proj_out_lora_a = self.proj_out_lora_a(indices)
        proj_out_lora_b = self.proj_out_lora_b(indices)

        # feedforward, but with expert loras chosen by pk

        # project in

        hidden = self.proj_in(x)

        lora_in_hidden = einsum(x, proj_in_lora_a, 'b n d, b n h k d -> b n h k')
        lora_in_hidden = lora_in_hidden * self.score_activation(scores)
        lora_in_hidden = einsum(lora_in_hidden, proj_in_lora_b, 'b n h k, b n h k d -> b n d')

        hidden = hidden + lora_in_hidden

        # gelu and dropout

        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)

        # project out

        out = self.proj_out(hidden)

        lora_out_hidden = einsum(hidden, proj_out_lora_a, 'b n d, b n h k d -> b n h k')
        lora_out_hidden = lora_out_hidden * self.score_activation(scores)
        lora_out_hidden = einsum(lora_out_hidden, proj_out_lora_b, 'b n h k, b n h k d -> b n d')

        out = out + lora_out_hidden

        return out
