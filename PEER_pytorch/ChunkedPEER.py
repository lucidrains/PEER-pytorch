import torch
from torch.nn import Module

from functools import partial
from torch.utils.checkpoint import checkpoint

from PEER_pytorch.PEER import PEER

class ChunkedPEER(Module):
    def __init__(
        self,
        peer: PEER,
        seq_chunk_size: int = 128
    ):
        super().__init__()
        self.peer = peer
        self.seq_chunk_size = seq_chunk_size

    def forward(
        self,
        x
    ):
        peer = self.peer

        if self.training and x.requires_grad:
            peer = partial(checkpoint, peer)            

        out = []
        for chunk in x.split(self.seq_chunk_size, dim = 1):
            chunk_out = peer(chunk)
            out.append(chunk_out)

        return torch.cat(out, dim = 1)

# quick test

if __name__ == '__main__':
    peer = PEER(dim = 512, heads = 8).cuda()

    peer = ChunkedPEER(peer)

    x = torch.randn(1, 1024, 512).cuda().requires_grad_()

    out = peer(x) + x

    out.sum().backward()
