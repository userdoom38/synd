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
Last updated: 2023-05-11
"""

import contextlib
import numpy as np
import torch

from synd.typing import *

# for information on context and random number generator management
# https://github.com/sdv-dev/CTGAN/blob/master/ctgan/synthesizers/base.py

@contextlib.contextmanager
def set_rng_states(
    state: Union[np.random.RandomState, torch.Generator],
    func: Callable,
):

    original_np_state = np.random.get_state()
    original_torch_state = torch.get_rng_state()

    np_rng, torch_rng, = state

    try:
        yield
    finally:

        curr_np_rng = np.random.RandomState()
        curr_np_rng.set_state(np_rng.get_state())
        curr_torch_rng = torch.Generator()
        curr_torch_rng.set_state(torch_rng.get_state())

        func((
            curr_np_rng,
            curr_torch_rng,
        ))

        np.random.set_state(original_np_state)
        torch.set_rng_state(original_torch_state)

def random_state(func: Callable) -> Any:
    def wrapper(self, *args, **kwargs):
        if self._rng_state is None:
            return func(self, *args, **kwargs)

        else:
            with set_rng_states(self._rng_state, self.set_rng_state):
                return func(self, *args, **kwargs)

    return wrapper

