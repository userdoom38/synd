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
Last updated: 2023-05-10
"""

from __future__ import annotations

import torch
import torch.nn as nn

import builtins
from typing import Dict

Integer = builtins.int
String = builtins.str

class Residual(nn.Module):
    """ Residual block for a neural network. """

    def __init__(self,
        in_dim: Integer,
        out_dim: Integer,
        *,
        device: String = 'cpu',
        **kwargs: Dict,
    ):
        super(Residual, self).__init__()

        self._in_dim = in_dim
        self._out_dim = out_dim
        self._device = device
        self._name = name

        sequence = [
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
        ]

        affine = nn.Sequential(*sequence)
        self._affine = affine

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self._affine(x), x], dim=1)

