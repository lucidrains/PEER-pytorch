import torch
from torch import nn
from torch.nn import Module

import einx
from einops.layers.torch import Rearrange
from einops import einsum

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# main class

class PK(Module):
    def __init__(
        self,
        dim,
        *,
        heads = 8,
        dim_key = None,
        num_keys = 1_000,
        product_keys = 2,
        product_key_topk = None,
        final_topk = 16,
        num_experts_per_head = 16
    ):
        """
        einops notation
        b - batch
        n - sequence
        d - dimension
        h - heads
        p - product keys
        k - number of keys
        """

        super().__init__()
        assert (dim % 2) == 0
        dim_key = default(dim_key, dim // 2)

        self.to_queries = nn.Sequential(
            nn.Linear(dim, dim_key * product_keys * heads, bias = False),
            Rearrange('b n (p h d) -> p b n h d', h = heads, p = product_keys)
        )

        self.num_keys = num_keys
        self.product_keys = product_keys

        self.keys = nn.Parameter(torch.zeros(product_keys, num_keys, heads, dim_key))
        nn.init.normal_(self.keys, std = 0.02)

        product_key_topk = default(product_key_topk, final_topk)
        assert final_topk <= (product_key_topk ** product_keys)

        self.topk = product_key_topk
        self.final_topk = final_topk

        # the maximum index, or the total space being indexed into

        self.max_index = int(num_keys ** product_keys)

    def forward(
        self,
        x,
        softmax_scores = False
    ):

        queries = self.to_queries(x)

        sim = einsum(queries, self.keys, 'p b n h d, p k h d -> p b n h k')

        scores, indices = sim.topk(self.topk, dim = -1)

        # cartesian product indices

        strides = self.num_keys ** torch.arange(self.product_keys, device = x.device)
        indices = einx.multiply('p ..., p -> p ...', indices, strides)

        index, *rest_indices = indices

        for rest_index in rest_indices:
            index = einx.add('... i, ... j -> ... (i j)', index, rest_index)

        # cartesian product score

        score, *rest_scores = scores

        for rest_score in rest_scores:
            score = einx.add('... i, ... j -> ... (i j)', score, rest_score)

        final_scores, final_indices = score, index

        # final topk

        final_scores, pk_indices = final_scores.topk(self.final_topk, dim = -1)

        final_indices = final_indices.gather(-1, pk_indices)

        if softmax_scores:
            final_scores = final_scores.softmax(dim = -1)

        return final_scores, final_indices


if __name__ == '__main__':

    pk = PK(
        dim = 512,
        num_keys = 100,
        final_topk = 10,
        product_keys = 3
    )

    x = torch.randn(2, 1024, 512)
    score, indices = pk(x)

    assert score.shape == (2, 1024, 8, 10)
    assert indices.shape == (2, 1024, 8, 10)
