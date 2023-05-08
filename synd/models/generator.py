"""
MIT License

Copyright (c) 2023 Wilhelm Ågren

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

File created: 2023-05-09
Last updated: 2023-05-09
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .residual import Residual

import builtins
from typing import Dict, List

Integer = builtins.int
Float = builtins.float
String = builtins.str

class Generator(nn.Module):
    """ Generator (decoder) neural network for GAN. """
    
    def __init__(self, emb_dim: Integer, gen_dim: List[Integer], data_dim: Integer,
                 device: String = 'cpu', name: String = 'Generator', **kwargs: Dict):
        super(Generator, self).__init__()

        self._emb_dim = emb_dim
        self._gen_dim = gen_dim
        self._data_dim = data_dim 
        self._device = device
        self._name = name

        dims = [emb_dim] + gen_dim
        sequence = [
            Residual(in_dim, out_dim)
            for in_dim, out_dim in zip(dims[:-1], dims[1:])
        ]

        sequence.append(nn.Linear(dims[-1], data_dim))
        decoder = nn.Sequential(*sequence)
        self._decoder = decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._decoder(x)
        
