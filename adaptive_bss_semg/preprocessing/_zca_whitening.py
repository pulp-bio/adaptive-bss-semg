"""Function and class implementing the ZCA whitening algorithm.


Copyright 2022 Mattia Orlandi

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

import logging
import warnings
from math import sqrt
from typing import overload

import numpy as np
import torch

from ._abc_whitening import WhiteningModel


@overload
def zca_whitening(
    x: np.ndarray,
    solver: str = "svd",
    device: torch.device | None = None,
) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
    """Function performing ZCA whitening.

    Parameters
    ----------
    x : ndarray
        Signal with shape (n_channels, n_samples).
    solver : {"svd", "eigh"}, default="svd"
        The solver used for whitening, either "svd" (default) or "eigh".
    device : device or None, default=None
        Torch device.

    Returns
    -------
    ndarray
        Whitened signal with shape (n_channels, n_samples).
    Tensor
        Estimated mean vector.
    Tensor
        Estimated whitening matrix.
    """


@overload
def zca_whitening(
    x: torch.Tensor,
    solver: str = "svd",
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Function performing ZCA whitening.

    Parameters
    ----------
    x : Tensor
        Signal with shape (n_channels, n_samples).
    solver : {"svd", "eigh"}, default="svd"
        The solver used for whitening, either "svd" (default) or "eigh".
    device : device or None, default=None
        Torch device.

    Returns
    -------
    Tensor
        Whitened signal with shape (n_channels, n_samples).
    Tensor
        Estimated mean vector.
    Tensor
        Estimated whitening matrix.
    """


def zca_whitening(
    x: np.ndarray | torch.Tensor,
    solver: str = "svd",
    device: torch.device | None = None,
) -> tuple[np.ndarray | torch.Tensor, torch.Tensor, torch.Tensor]:
    """Function performing ZCA whitening.

    Parameters
    ----------
    x : ndarray or Tensor
        Signal with shape (n_channels, n_samples).
    solver : {"svd", "eigh"}, default="svd"
        The solver used for whitening, either "svd" (default) or "eigh".
    device : device or None, default=None
        Torch device.

    Returns
    -------
    ndarray or Tensor
        Whitened signal with shape (n_channels, n_samples).
    Tensor
        Estimated mean vector.
    Tensor
        Estimated whitening matrix.
    """
    whiten_model = ZCAWhitening(solver, device)
    x_w = whiten_model.fit_transform(x)

    return x_w, whiten_model.mean_vec, whiten_model.white_mtx


class ZCAWhitening(WhiteningModel):
    """Class implementing ZCA whitening.

    Parameters
    ----------
    solver : {"svd", "eigh"}, default="svd"
        The solver used for whitening, either "svd" (default) or "eigh".
    device : device or None, default=None
        Torch device.

    Attributes
    ----------
    _solver : str
        The solver used for whitening, either "svd" (default) or "eigh".
    _device : device or None
        Torch device.
    """

    def __init__(self, solver: str = "svd", device: torch.device | None = None) -> None:
        assert solver in ("svd", "eigh"), 'The solver must be either "svd" or "eigh".'

        logging.info(f'Instantiating ZCAWhitening using "{solver}" solver.')

        self._solver: str = solver
        self._device: torch.device | None = device

        self._mean_vec: torch.Tensor | None = None
        self._white_mtx: torch.Tensor | None = None

    @property
    def mean_vec(self) -> torch.Tensor | None:
        """Tensor or None: Property for getting the estimated mean vector."""
        return self._mean_vec

    @property
    def white_mtx(self) -> torch.Tensor:
        """Tensor or None: Property for getting the estimated whitening matrix."""
        return self._white_mtx

    def fit(self, x: np.ndarray | torch.Tensor) -> WhiteningModel:
        """Fit the whitening model on the given signal.

        Parameters
        ----------
        x :  ndarray or Tensor
            Signal with shape (n_channels, n_samples).

        Returns
        -------
        WhiteningModel
            The fitted whitening model.
        """
        # Fit the model and return self
        self._fit_transform(x)
        return self

    @overload
    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        """Fit the whitening model on the given signal and return the whitened signal.

        Parameters
        ----------
        x : ndarray
            Signal with shape (n_channels, n_samples).

        Returns
        -------
        ndarray
            Whitened signal with shape (n_components, n_samples).
        """

    @overload
    def fit_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Fit the whitening model on the given signal and return the whitened signal.

        Parameters
        ----------
        x : Tensor
            Signal with shape (n_channels, n_samples).

        Returns
        -------
        Tensor
            Whitened signal with shape (n_components, n_samples).
        """

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
        # Fit the model and return result
        return self._fit_transform(x)

    @overload
    def transform(self, x: np.ndarray) -> np.ndarray:
        """Whiten the given signal using the fitted whitening model.

        Parameters
        ----------
        x : ndarray
            Signal with shape (n_channels, n_samples).

        Returns
        -------
        ndarray
            Whitened signal with shape (n_components, n_samples).
        """

    @overload
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Whiten the given signal using the fitted whitening model.

        Parameters
        ----------
        x : Tensor
            Signal with shape (n_channels, n_samples).

        Returns
        -------
        Tensor
            Whitened signal with shape (n_components, n_samples).
        """

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
        assert (
            self._mean_vec is not None and self._white_mtx is not None
        ), "Mean vector or whitening matrix are null, fit the model first."

        # Convert input to Tensor
        is_numpy = isinstance(x, np.ndarray)
        x_t = torch.from_numpy(x).to(self._device) if is_numpy else x.to(self._device)

        # Center signal
        x_t -= self._mean_vec
        # Whiten signal
        x_w_t = self._white_mtx @ x_t

        if is_numpy:
            return x_w_t.cpu().numpy()
        else:
            return x_w_t

    @overload
    def _fit_transform(self, x: np.ndarray) -> np.ndarray:
        """Helper method for fit and fit_transform."""

    @overload
    def _fit_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Helper method for fit and fit_transform."""

    def _fit_transform(self, x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """Helper method for fit and fit_transform."""
        is_numpy = isinstance(x, np.ndarray)

        x_t = torch.from_numpy(x).to(self._device) if is_numpy else x.to(self._device)
        n_samp = x_t.size(1)
        self._mean_vec = x_t.mean(dim=1, keepdim=True)
        x_t -= self._mean_vec

        if self._solver == "svd":
            e, d, _ = torch.linalg.svd(x_t, full_matrices=False)

            d_mtx = torch.diag(1.0 / d) * sqrt(n_samp - 1)
        elif self._solver == "eigh":
            d, e = torch.linalg.eigh(torch.cov(x_t))

            # Improve numerical stability
            eps = torch.finfo(d.dtype).eps
            degenerate_idx = torch.lt(d, eps).nonzero()
            if torch.any(degenerate_idx):
                warnings.warn(
                    f'Some eigenvalues are smaller than epsilon ({eps:.3e}), try using "SVD" solver.'
                )
            d[degenerate_idx] = eps

            sort_idx = torch.argsort(d, descending=True)
            d, e = d[sort_idx], e[:, sort_idx]

            d_mtx = torch.diag(1.0 / torch.sqrt(d))
        else:
            raise NotImplementedError("Unknown solver.")

        e *= torch.sign(e[0])  # guarantee consistent sign

        self._white_mtx = e @ d_mtx @ e.T
        x_w_t = self._white_mtx @ x_t

        if is_numpy:
            return x_w_t.cpu().numpy()
        else:
            return x_w_t
