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

File created: 2023-05-12
Last updated: 2023-05-12
"""

import logging
import unittest
import pandas as pd
import numpy as np
import torch

from sdv.metadata import SingleTableMetadata
from synd.models import CTGAN
from synd.datasets import SingleTable

log = logging.getLogger(__name__)

class TGANTest(unittest.TestCase):
    """ Unit-tests for the TGAN model. """

    def test_fit_sample(self):
        n = 1000
        log.debug(f'creating mocked data, {n=}')
        data = pd.DataFrame({
            'id': np.arange(n),
            'age': np.random.randint(low=18, high=100, size=(n, )),
            'salary': np.random.uniform(low=1000, high=100000, size=(n, )),
            'gender': np.random.choice(['male', 'female', 'none'], size=(n, )),
        })

        discrete_columns = [
            'id',
            'gender',
        ]

        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data)

        log.debug('fitting dataset')
        dataset = SingleTable(data, metadata.to_dict(), discrete_columns=discrete_columns)
        dataset.fit()

        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        model = CTGAN(batch_size=200, device=device)

        epochs = 2
        log.debug(f'fitting model for {epochs=}')
        model.fit(dataset, epochs=epochs)

        samples = 243
        log.debug(f'sampling {samples=}')
        fake = model.sample(samples)

        self.assertTrue(len(fake) == samples)
        self.assertTrue(len(fake.columns) == len(data.columns))
        self.assertTrue(getattr(model, '_critic', None) is None)

    def test_init_error(self):
        self.assertRaises(
            AssertionError,
            CTGAN,
            batch_size=17,
        )

