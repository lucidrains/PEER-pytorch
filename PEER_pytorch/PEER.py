import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList

# helper functions

def exists(v):
    return v is not None

# main class

class PEER(Module):
    def __init__(
        self,
        dim
    ):
        super().__init__()

    def forward(
        self,
        x
    ):
        return x
