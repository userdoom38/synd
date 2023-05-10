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
Last updated: 2023-05-10
"""

import logging
import pandas as pd

from ctgan.data_transformer import DataTransformer
from ctgan.data_sampler import DataSampler

from .base import Dataset

import builtins
from typing import (
    Union,
    Dict,
    List,
    NoReturn,
    Optional,
)

Integer = builtins.int
Float = builtins.float
String = builtins.str
Boolean = builtins.bool

logger = logging.getLogger(__name__)

class SingleTable(Dataset):
    """ """

    def __init__(self,
        data: pd.DataFrame,
        metadata: Dict,
        *,
        log_frequency: Optional[Boolean] = None,
        discrete_columns: Optional[List[String]] = None,
        max_clusters: Integer = 10,
        weight_threshold: Float = 0.005,
        **kwargs: Dict,
    ):

        if log_frequency is None:
            log_frequency = True 

        if discrete_columns is None:
            discrete_columns = []

        self._data = data
        self._metadata = metadata

        self._log_frequency = log_frequency
        self._discrete_columns = discrete_columns
        self._max_clusters = max_clusters
        self._weight_threshold = weight_threshold

    def __len__(self) -> Integer:
        return self._data.shape[0]

    def fit(self) -> NoReturn:
        """ 
        Create a `DataTransformer` and fit it with the provided data.
        Transforms the data and creates a `DataSampler` based on it.

        The `DataSampler` is used during training and generation,
        and the `DataTransformer` is used after the sampling process
        to transform the generated data to the correct output format.
        """

        transformer = DataTransformer(
            max_clusters=self._max_clusters,
            weight_threshold=self._weight_threshold,
        )

        transformer.fit(self._data, self._discrete_columns)
        data = transformer.transform(self._data)

        sampler = DataSampler(
            data,
            transformer.output_info_list,
            self._log_frequency,
        )

        self._transformer = transformer
        self._sampler = sampler

    def is_fitted(self) -> Boolean:
        """ 
        If the transformer has been fitted, the object will have the attribute 
        `dataframe`. Then return `True` if it is fitted, else `False`.
        """
        return hasattr(self._transformer, 'dataframe')
    
    @property
    def transformer(self) -> Optional[DataTransformer]:
        return getattr(self, '_transformer', None)

    @property
    def sampler(self) -> Optional[DataSampler]:
        return getattr(self, '_sampler', None)

