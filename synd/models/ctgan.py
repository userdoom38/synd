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

import logging
import random

import numpy as np
import pandas as pd

import torch
from torch import optim

from ctgan.data_sampler import DataSampler
from ctgan.data_transformer import DataTransformer

from synd.datasets import SingleTable 
from .base import Synthesizer
from .critic import Critic
from .generator import Generator

import builtins
from typing import (
    Tuple,
    Dict,
    Union,
)

Integer = builtins.int
Float = builtins.float
String = builtins.str
Boolean = builtins.bool

logger = logging.getLogger(__name__)

class CTGAN(Synthesizer):
    """ Conditional Tabular Generative Adversarial Network. """
    
    def __init__(self, *,
        embedding_dim: Integer = 128, 
        generator_dims: Tuple[Integer, ...] = (256, 256), 
        critic_dims: Tuple[Integer, ...] = (256, 256),
        generator_lr: Float = 2e-4,
        critic_lr: Float = 2e-4,
        generator_decay: Float = 1e-6,
        critic_decay: Float = 1e-6, 
        generator_betas: Tuple[Float, Float] = (0.5, 0.9), 
        critic_betas: Tuple[Float, Float] = (0.5, 0.9),
        batch_size: Integer = 500, 
        critic_steps: Integer = 1, 
        pac: Integer = 10,
        device: Union[String, torch.device] = 'cpu',
        **kwargs: Dict,
    ):

        assert not batch_size % 2

        self._embedding_dim = embedding_dim
        self._generator_dims = generator_dims
        self._critic_dims = critic_dims

        self._generator_lr = generator_lr
        self._generator_decay = generator_decay
        self._generator_betas = generator_betas
        self._critic_lr = critic_lr
        self._critic_decay = critic_decay
        self._critic_betas = critic_betas

        self._batch_size = batch_size
        self._critic_steps = critic_steps
        self._pac = pac
        self._device = device

    def fit(self,
        dataset: SingleTable,
        *,
        discrete_columns: List[String, ...],
        epochs: Integer = 100,
        **kwargs: Dict,
    ):
        """ Train the Wasserstein PacGAN on the provided training data. """

        if not dataset.is_fitted():
            dataset.fit()

        self._transformer = dataset.transformer()
        self._sampler = dataset.sampler()

        data_dim = self._transformer.output_dimensions
        
        generator = Generator(
            self._embedding_dim + self._sampler.dim_cond_vec(),
            self._generator_dims,
            data_dim,
        ).to(self._device)

        critic = Critic(
            data_dim + self._sampler.dim_cond_vec(),
            self._critic_dims,
            pac=self._pac,
        ).to(self._device)

        optimG = optim.Adam(
            generator.parameters(), lr=self._generator_lr,
            betas=self._generator_betas, weight_decay=self._generator_decay,
        )

        optimC = optim.Adam(
            critic.parameters(), lr=self._critic_lr,
            betas=self._critic_betas, weight_decay=self._critic_decay,
        )

        self._generator = generator
        self._critic = critic

        steps_per_epoch = max(len(data) // self._batch_size, 1)
        for i in range(epochs):
            for step in range(steps_per_epoch):
                for n in range(self._critic_steps):
                    pass

