"""
MIT License

Copyright (c) 2023 Wilhelm Ågren

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
coalphaes of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
coalphaes or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

File created: 2023-05-08
Last updated: 2023-05-09
"""

from __future__ import annotations

import torch
import torch.nn as nn

import builtins
from typing import (
    Optional,
    Union,
    Tuple,
    List,
    Dict,
    NoReturn,
)

Integer = builtins.int
Float = builtins.float
String = builtins.str

class Critic(nn.Module):
    """
    Critic neural network for Wasserstein PacGAN.

    [1] Martin Arjovsky et al., 2017 `https://arxiv.org/abs/1701.07875`
    [2] Zinan Lin et al., 2017 `https://arxiv.org/abs/1712.04086`
    [3] Ishaan Gulrajani., 2017 `https://arxiv.org/abs/1704.00028`
    """

    def __init__(self, input_dim: Integer, critic_dims: List[Integer],
                 negative_slope: Float = 0.2, dropout: Float = 0.5,
                 pac: Integer = 10, device: String = 'cpu',
                 name: String = 'Critic', **kwargs: Dict):
        super(Critic, self).__init__()

        self._input_dim = input_dim
        self._critic_dims = critic_dims
        self._negative_slope = negative_slope 
        self._dropout = dropout
        self._pac = pac
        self._device = device
        self._name = name

        dims = [input_dim * pac] + critic_dims
        sequence = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            sequence.extend([
                nn.Linear(in_dim, out_dim),
                nn.LeakyReLU(negative_slope),
                nn.Dropout(dropout),
            ])

        encoder = nn.Sequential(*sequence)
        self._encoder = encoder 

    def calc_gradient_penalty(self, real: torch.Tensor, fake: torch.Tensor,
                              lambda_: Integer = 10, **kwargs: Dict) -> torch.Tensor:
        """ Compute the gradient penalty according to [3]. """
        real_size = real.size
        fake_size = fake.size

        alpha = torch.rand(real_size(0) // self._pac, 1, 1).to(self._device)
        alpha = alpha.repeat(1, self._pac, real_size(1)).view(
            -1, real_size(1),
        )

        # uniformly sampling along straight lines between pairs of points between
        # data distribution and generator distribution, with norm 1 coupling
        samples = alpha * real + ((1 - alpha) * fake)

        enc_samples = self(samples)

        gradients = torch.autograd.grad(
            outputs=enc_samples, inputs=samples,
            grad_outputs=torch.ones(enc_samples.size()).to(self._device),
            create_graph=True, retain_graph=True, only_inputs=True,
        )[0]

        gradients = gradients.view(-1, self._pac * real_size(1)).norm(2, dim=1) - 1
        gradient_penalty = lambda_ * ((gradients) ** 2).mean()

        return gradient_penalty

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.size()[0] % self._pac == 0, \
        f'Batch size of input tensor must be divisible by the pac size. ' \
        f'Your shape is {x.shape}, and {x.size()[0]} is not divisible by {self._pac}.'

        return self._encoder(x.view(-1, self._input_dim * self._pac))

    @classmethod
    def no_pac(cls, *args, **kwargs) -> Critic:
        return cls(*args, pac=1, **kwargs)

