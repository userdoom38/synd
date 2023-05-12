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

from __future__ import annotations

import pandas as pd

from datetime import datetime

from ctgan.data_sampler import DataSampler
from ctgan.data_transformer import DataTransformer

from synd.models import Synthesizer
from synd.utils import create_timestamp
from synd.typing import *

class SampleSet(object):
    """ """
    def __init__(self, *args, **kwargs):
        for kwarg, value in kwargs.items():
            setattr(self, '_' + kwarg, value)

    @classmethod
    def single_table(cls,
        name: String,
        data: pd.DataFrame,
        metadata: Dict,
        model: Synthesizer,
        training_data: pd.DataFrame,
        data_sampler: DataSampler,
        data_transformer: DataTransformer,
        timestamp: String,
        model_name: String,
        dataset_name: String,
        **kwargs: Dict,
    ) -> SampleSet:
        return cls(
            name=name,
            data=data,
            metadata=metadata,
            model=model,
            training_data=training_data,
            data_sampler=data_sampler,
            data_transformer=data_transformer,
            timestamp=timestamp,
            model_name=model_name,
            dataset_name=dataset_name,
            **kwargs,
        )

    def save_sampled_data(self):
        pass

    def save(self):
        pass

