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

File created: 2023-05-10
Last updated: 2023-05-11
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from synd.typing import *

class Synthesizer(object):
    """ Base synthesizer class for all synthetic data generator models. """
    def __init__(self, *args, **kwargs):
        self.set_rng_state(**kwargs)

    def fit(self, *args, **kwargs):
        raise NotImplementedError(
            f'No training method is implemented for {self.__class__}. '
            f'Please implement a training process for synthesizer.'
        )

    def sample(self, *args, **kwargs):
        raise NotImplementedError(
            f'No sampling method is implemented for {self.__class__}. '
            f'Please implement a sampling process for the synthesizer.'
        )

    def to(self, device: Union[String, torch.device]) -> Synthesizer:
        """ Move all `torch.nn` attributes of `self` to the specified `device`. """
        for attr in self.__dict__.values():
            if isinstance(attr, nn.Module):
                attr.to(device)

        return self

    @classmethod
    def from_pretrained(cls, path: String) -> Synthesizer:
        model = torch.load(path)
        device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu',
        )

        model.to(device)
        return model

    def set_rng_state(self,
        state: Union[Integer, Tuple[np.random.RandomState, torch.Generator]] = 1337,
        **kwargs: Dict,
    ):
        """ Set the state for all relevant random number generators. """
        # see below link for random number generation management
        # https://github.com/sdv-dev/CTGAN/blob/master/ctgan/synthesizers/base.py

        if isinstance(state, int):
            self._rng_state = (
                np.random.RandomState(seed=state),
                torch.Generator().manual_seed(state),
            )
        elif (
            isinstance(state, tuple) and
            isinstance(state[0], np.random.RandomState) and
            isinstance(state[1], torch.Generator)
        ):
            self._rng_state = state
        else:
            raise TypeError(
                f'The provided state should either be `int` or (`np.random.RandomState`, '
                f'`torch.Generator`). Function got: {state}.'
            )

