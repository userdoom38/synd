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

File created: 2023-05-13
Last updated: 2023-05-13
"""

from __future__ import annotations

import logging
import pandas as pd
import numpy as np

from dataclasses import dataclass

from sklearn.preprocessing import OneHotEncoder
from sklearn.mixture import BayesianGaussianMixture

from synd.typing import *

log = logging.getLogger(__name__)

@dataclass(frozen=True)
class ActivationFnInfo:
    dim: Integer
    activation_fn: String

@dataclass(frozen=True)
class ColumnTransformInfo:
    name: String
    type_: String
    info: List[ActivationFnInfo]
    dims: Integer

class TabularTransformer(object):
    """ """

    def __init__(self, *,
        max_bgm_clusters=10,
        weight_threshold=0.005,
        **kwargs: Dict,
    ):
        self._max_bgm_clusters = max_bgm_clusters
        self._weight_threshold = weight_threshold

        self._fit_func = {
            'discrete': self._fit_discrete,
            'continuous': self._fit_continuous,
        }

    def fit(self,
        data: pd.DataFrame,
        *,
        discrete_columns: Tuple[String, ...] = (),
        **kwargs: Dict,
    ):
        """ """
        transform_info = []
        column_info = []

        col_dtypes = data.infer_objects().dtypes

        for column in data.columns:
            column_transform_info = self._fit_func[
                'discrete' if column in discrete_columns else 'continuous'
            ](data[[column]])
            column_info.append(column_transform_info.info)
            transform_info.append(column_transform_info)

        self._transform_info = transform_info
        self._column_info

    @staticmethod
    def _fit_discrete(
        data: pd.DataFrame,
        *,
        handle_unknown: String = 'error',
        **kwargs: Dict,
    ) -> ColumnTransformInfo:
        """ """

        column = data.columns[0]

        ohe = OneHotEncoder(handle_unknown=handle_unknown)
        ohe.fit(data)

        n_categories = ohe.n_features_in_
        self._ohe = ohe

        return ColumnTransformInfo(
            name=column,
            type='discrete',
            info=[ActivationFnInfo(n_categories, 'softmax')],
            dims=n_categories,
        )

    @staticmethod
    def _fit_continuous(
        data: pd.DataFrame,
        *,
        n_init: Integer = 1,
        init_params: String = 'kmeans++',
        weight_concentration_prior_type: String = 'dirichlet_distribution',
        **kwargs: Dict,
    ) -> ColumnTransformInfo:
        """ """

        column = data.columns[0]

        bgm = BayesianGaussianMixture(
            n_components=self._max_bqm_clusters,
            n_init=n_init,
            init_params=init_params,
            weight_concentration_prior_type=weight_concentration_prior_type,
            random_state=get_rng_state()['numpy'],
        )
        bgm.fit(data.reshape(-1, 1))

        n_components = sum(bgm.weights_ > self._weight_threshold)
        self._bgm = bgm

        return ColumnTransformInfo(
            name=column,
            type_='continuous',
            info=[
                ActivationFnInfo(1, 'tanh'),
                ActivationFnInfo(n_components, 'softmax'),
            ],
            dims=1 + n_components,
        )

