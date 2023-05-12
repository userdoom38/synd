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

File created: 2023-05-11
Last updated: 2023-05-12
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch import optim

from datetime import datetime

from ctgan.data_sampler import DataSampler
from ctgan.data_transformer import DataTransformer

from synd.datautil import SampleSet
from synd.datasets import SingleTable
from synd.utils import random_state, create_timestamp
from synd.typing import *
from .base import Synthesizer
from .critic import Critic
from .generator import Generator

log = logging.getLogger(__name__)

class TGAN(Synthesizer):
    """ Tabular Generative Adversarial Network,
    (with packing, Wasserstein loss, and gradient penalty). """

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
        lambda_: Integer = 10,
        device: Union[String, torch.device] = 'cpu',
        **kwargs: Dict,
    ):
        super(TGAN, self).__init__()

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
        self._lambda_ = lambda_
        self._device = device

    def _apply_activate(self, data):
        data_t = []
        st = 0
        for column_info in self._transformer.output_info_list:
            for span_info in column_info:
                if span_info.activation_fn == 'tanh':
                    ed = st + span_info.dim
                    data_t.append(torch.tanh(data[:, st:ed]))
                    st = ed
                elif span_info.activation_fn == 'softmax':
                    ed = st + span_info.dim
                    data_t.append(F.gumbel_softmax(
                        data[:, st:ed],
                        tau=0.2,
                        hard=False,
                        eps=1e-10,
                        dim=-1,
                    ))
                    st = ed
                else:
                    raise ValueError(
                        f'Unexpected activation function {span_info.activation_fn}.',
                    )

        return torch.cat(data_t, dim=1)

    @random_state
    def fit(self,
        dataset: SingleTable,
        *,
        discard_critic: Boolean = True,
        epochs: Integer = 100,
        **kwargs: Dict,
    ):
        """ Train the Wasserstein TGAN with packing and gradient penalty. """
        
        if not dataset.is_fitted():
            log.debug('fitting `SingleTable` dataset')
            dataset.fit()

        self._transformer = dataset.transformer
        sampler = dataset.sampler

        data_dim = self._transformer.output_dimensions

        generator = Generator(
            self._embedding_dim,
            self._generator_dims,
            data_dim,
        ).to(self._device)

        critic = Critic(
            data_dim,
            self._critic_dims,
            pac=self._pac,
        ).to(self._device)

        optimizer_G = optim.Adam(
            generator.parameters(),
            lr=self._generator_lr,
            betas=self._generator_betas,
            weight_decay=self._generator_decay,
        )

        optimizer_C = optim.Adam(
            critic.parameters(),
            lr=self._critic_lr,
            betas=self._critic_betas,
            weight_decay=self._critic_decay,
        )

        self._generator = generator

        mean = torch.zeros(
            self._batch_size,
            self._embedding_dim,
            device=self._device,
        )
        std = mean + 1

        print(f'Epoch\t\tG loss\t\tC loss')
        print('=' * 45)

        steps_per_epoch = max(len(dataset) // self._batch_size, 1)
        for i in range(epochs):
            for step in range(steps_per_epoch):
                for n in range(self._critic_steps):
                    real = torch.Tensor(sampler.sample_data(
                        self._batch_size, None, None,
                    ).astype(np.float32)).to(self._device)

                    z = torch.normal(mean, std)
                    fake = self._generator(z)
                    fakeact = self._apply_activate(fake)

                    y_real = critic(real)
                    y_fake = critic(fakeact)

                    # calculate gradient penalty
                    real_size = real.size
                    fake_size = fake.size

                    alpha = torch.rand(real_size(0) // self._pac,
                        1, 1, device=self._device,
                    )
                    alpha = alpha.repeat(1, self._pac, real_size(1)).view(
                        -1, real_size(1),
                    )

                    samples = alpha * real + ((1 - alpha) * fake)
                    enc_samples = critic(samples)

                    grad_outputs = torch.ones(enc_samples.size()).to(self._device)
                    gradients = torch.autograd.grad(
                        inputs=samples,
                        outputs=enc_samples,
                        grad_outputs=grad_outputs,
                        create_graph=True,
                        retain_graph=True,
                        only_inputs=True,
                    )[0]

                    gradients = gradients.view(-1, self._pac * real_size(1))
                    gradient_penalty = self._lambda_ * ((gradients.norm(2, dim=1) - 1) ** 2).mean()

                    loss_c = -(torch.mean(y_real) - torch.mean(y_fake)) + gradient_penalty

                    optimizer_C.zero_grad()
                    loss_c.backward()
                    optimizer_C.step()

                z = torch.normal(mean, std)
                fake = self._generator(z)
                fakeact = self._apply_activate(fake)
                y_fake = critic(fakeact)

                loss_g = -torch.mean(y_fake)

                optimizer_G.zero_grad()
                loss_g.backward()
                optimizer_G.step()

            loss_c = loss_c.detach().cpu()
            loss_g = loss_g.detach().cpu()
            print(
                f'{i + 1:02}\t\t{loss_g:.4f}\t\t{loss_c:.4f}',
                flush=True,
            )

        if not discard_critic:
            log.debug('saving trained critic')
            self._critic = critic

    @random_state
    def sample(self,
        n_samples: Integer,
        dataset: SingleTable,
        *,
        sampleset_name: Optional[String] = None,
        **kwargs: Dict,
    ) -> SampleSet:
        """ Sample synthetic data from the trained generator. """

        if sampleset_name is None:
            sampleset_name = SampleSet.__name__ + create_timestamp()

        batch_sizes = [self._batch_size for _ in range(n_samples // self._batch_size)]
        batch_sizes += [n_samples - sum(batch_sizes)]

        samples = []
        with torch.no_grad():
            self._generator.eval()
            for batch_size in batch_sizes:
                mean = torch.zeros(batch_size, self._embedding_dim)
                std = mean + 1
                z = torch.normal(mean, std).to(self._device)

                fake = self._generator(z)
                fakeact = self._apply_activate(fake)
                samples.append(fakeact.cpu().numpy())

        samples = self._transformer.inverse_transform(np.concatenate(samples, axis=0))

        return SampleSet.single_table(
            name=sampleset_name,
            data=samples,
            metadata=dataset.metadata,
            model=self,
            training_data=dataset.data,
            data_sampler=dataset.sampler,
            data_transformer=dataset.transformer,
            timestamp=create_timestamp(),
            model_name=self.__class__.__name__,
            dataset_name=dataset.name,
        )

