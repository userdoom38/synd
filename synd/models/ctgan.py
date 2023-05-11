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
Last updated: 2023-05-11
"""

from __future__ import annotations

import logging
import random

import numpy as np
import pandas as pd

import torch
from torch import optim
import torch.nn.functional as F

from ctgan.data_sampler import DataSampler
from ctgan.data_transformer import DataTransformer

from synd.datasets import SingleTable 
from synd.utils import random_state
from synd.typing import *
from .base import Synthesizer
from .critic import Critic
from .generator import Generator

log = logging.getLogger(__name__)

class CTGAN(Synthesizer):
    """ Conditional Tabular Generative Adversarial Network,
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
        super(CTGAN, self).__init__()

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
                    raise ValueError(f'Unexpected activation function {span_info.activation_fn}.')
        return torch.cat(data_t, dim=1)

    def _cond_loss(self, data, c, m):
        """Compute the cross entropy loss on the fixed discrete column."""
        loss = []
        st = 0
        st_c = 0
        for column_info in self._transformer.output_info_list:
            for span_info in column_info:
                if len(column_info) != 1 or span_info.activation_fn != 'softmax':
                    # not discrete column
                    st += span_info.dim
                else:
                    ed = st + span_info.dim
                    ed_c = st_c + span_info.dim
                    tmp = F.cross_entropy(
                        data[:, st:ed],
                        torch.argmax(c[:, st_c:ed_c], dim=1),
                        reduction='none'
                    )
                    loss.append(tmp)
                    st = ed
                    st_c = ed_c

        loss = torch.stack(loss, dim=1)  # noqa: PD013

        return (loss * m).sum() / data.size()[0]

    @random_state
    def fit(self,
        dataset: SingleTable,
        *,
        discard_critic: Boolean = True,
        epochs: Integer = 100,
        **kwargs: Dict,
    ):
        """ Train the Wasserstein PacGAN on the provided training data. """

        if not dataset.is_fitted():
            log.info(f'fitting {dataset}')
            dataset.fit()

        self._transformer = dataset.transformer
        self._sampler = dataset.sampler

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

        optimizer_G = optim.Adam(
            generator.parameters(), lr=self._generator_lr,
            betas=self._generator_betas, weight_decay=self._generator_decay,
        )

        optimizer_C = optim.Adam(
            critic.parameters(), lr=self._critic_lr,
            betas=self._critic_betas, weight_decay=self._critic_decay,
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
                    z = torch.normal(mean, std)
                    condvec = self._sampler.sample_condvec(self._batch_size)

                    if condvec is None:
                        c1, m1, col, opt, = (None, ) * 4
                        real = self._sampler.sample_data(
                            self._batch_size,
                            col,
                            opt,
                        )
                    else:
                        c1, m1, col, opt, = condvec
                        c1 = torch.Tensor(c1.astype(np.float32)).to(self._device)
                        m1 = torch.Tensor(m1.astype(np.float32)).to(self._device)
                        z = torch.cat([z, c1], dim=1)

                        perm = np.arange(self._batch_size)
                        np.random.shuffle(perm)

                        c2 = c1[perm]
                        real = self._sampler.sample_data(
                            self._batch_size,
                            col[perm],
                            opt[perm],
                        )

                    real = torch.Tensor(real.astype(np.float32)).to(self._device)
                    fake = self._generator(z)
                    fakeact = self._apply_activate(fake)

                    if c1 is not None:
                        fake_cat = torch.cat([fakeact, c1], dim=1)
                        real_cat = torch.cat([real, c2], dim=1)
                    else:
                        fake_cat = fakeact
                        real_cat = real

                    y_fake = critic(fake_cat)
                    y_real = critic(real_cat)

                    # calculate gradient penalty
                    # https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan_gp/wgan_gp.py
                    fake_size = fake_cat.size
                    real_size = real_cat.size

                    alpha = torch.rand(real_size(0) // self._pac,
                        1, 1, device=self._device,
                    )
                    alpha = alpha.repeat(1, self._pac, real_size(1)).view(
                        -1, real_size(1),
                    )

                    samples = alpha * real_cat + ((1 - alpha) * fake_cat)
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
                c = self._sampler.sample_condvec(self._batch_size)

                if condvec is None:
                    c1, m1, col, opt, = (None, ) * 4
                else:
                    c1, m1, col, opt, = condvec
                    c1 = torch.Tensor(c1.astype(np.float32)).to(self._device)
                    m1 = torch.Tensor(m1.astype(np.float32)).to(self._device)
                    z = torch.cat([z, c1], dim=1)

                fake = self._generator(z)
                fakeact = self._apply_activate(fake)

                if c1 is not None:
                    y_fake = critic(torch.cat([fakeact, c1], dim=1))
                else:
                    y_fake = critic(fakeact)

                if condvec is None:
                    cross_entropy = 0.0
                else:
                    cross_entropy = self._cond_loss(fake, c1, m1)

                loss_g = -torch.mean(y_fake) + cross_entropy

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
            log.info('saving trained critic')
            self._critic = critic

    @random_state
    def sample(self,
        n_samples: Integer,
        **kwargs: Dict,
    ) -> pd.DataFrame:
        """ Sample synthetic data from the trained generator. """

        batch_sizes = [self._batch_size for _ in range(n_samples // self._batch_size)]
        batch_sizes += [n_samples - sum(batch_sizes)]

        data = []
        with torch.no_grad():
            self._generator.eval()
            for batch_size in batch_sizes:
                mean = torch.zeros(batch_size, self._embedding_dim)
                std = mean + 1
                z = torch.normal(mean, std).to(self._device)

                condvec = self._sampler.sample_original_condvec(batch_size)

                if condvec is not None:
                    c1 = torch.Tensor(condvec.astype(np.float32)).to(self._device)
                    z = torch.cat([z, c1], dim=1)

                fake = self._generator(z)
                fakeact = self._apply_activate(fake)
                data.append(fakeact.cpu().numpy())

        data = np.concatenate(data, axis=0)
        return self._transformer.inverse_transform(data)

