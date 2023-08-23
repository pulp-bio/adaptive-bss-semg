"""Interface for whitening algorithms.


Copyright 2023 Mattia Orlandi

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch


class WhiteningModel(ABC):
    """Interface for performing whitening."""

    @property
    @abstractmethod
    def mean_vec(self) -> torch.Tensor | None:
        """Tensor or None: Property for getting the estimated mean vector."""

    @property
    @abstractmethod
    def white_mtx(self) -> torch.Tensor | None:
        """Tensor or None: Property for getting the estimated whitening matrix."""

    @abstractmethod
    def fit(self, x: np.ndarray | torch.Tensor) -> WhiteningModel:
        """Fit the whitening model on the given signal.

        Parameters
        ----------
        x : ndarray or Tensor
            Signal with shape (n_channels, n_samples).

        Returns
        -------
        Whitening
            The fitted whitening model.
        """

    @abstractmethod
    def fit_transform(self, x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """Fit the whitening model on the given signal and return the whitened signal.

        Parameters
        ----------
        x : ndarray or Tensor
            Signal with shape (n_channels, n_samples).

        Returns
        -------
        ndarray or Tensor
            Whitened signal with shape (n_channels, n_samples).
        """

    @abstractmethod
    def transform(self, x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """Whiten the given signal using the fitted whitening model.

        Parameters
        ----------
        x : ndarray or Tensor
            Signal with shape (n_channels, n_samples).

        Returns
        -------
        ndarray or Tensor
            Whitened signal with shape (n_channels, n_samples).
        """
