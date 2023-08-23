"""Function and class implementing the PCA whitening algorithm.


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

from adaptive_bss_semg.preprocessing import WhiteningModel


@overload
def pca_whitening(
    x: np.ndarray,
    n_pcs: int = -1,
    p_discard: float = 0,
    var_th: float = 1,
    solver: str = "svd",
    device: torch.device | None = None,
) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
    """Function performing PCA whitening.

    Parameters
    ----------
    x : ndarray
        Signal with shape (n_channels, n_samples).
    n_pcs : int, default=-1
        Number of components to be selected (if zero or negative, all components will be retained)).
    p_discard : float, default=0
        Proportion of components to be discarded; relevant if n_pcs is not specified.
    var_th : float, default=1
        Cut-off threshold for the variances; relevant if n_pcs and p_discard are not specified.
    solver : {"svd", "eigh"}, default="svd"
        The solver used for whitening, either "svd" (default) or "eigh".
    device : device or None, default=None
        Torch device.

    Returns
    -------
    ndarray
        Whitened signal with shape (n_components, n_samples).
    Tensor
        Estimated mean vector.
    Tensor
        Estimated whitening matrix.
    """


@overload
def pca_whitening(
    x: torch.Tensor,
    n_pcs: int = -1,
    p_discard: float = 0,
    var_th: float = 1,
    solver: str = "svd",
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Function performing PCA whitening.

    Parameters
    ----------
    x : Tensor
        Signal with shape (n_channels, n_samples).
    n_pcs : int, default=-1
        Number of components to be selected (if zero or negative, all components will be retained)).
    p_discard : float, default=0
        Proportion of components to be discarded; relevant if n_pcs is not specified.
    var_th : float, default=1
        Cut-off threshold for the variances; relevant if n_pcs and p_discard are not specified.
    solver : {"svd", "eigh"}, default="svd"
        The solver used for whitening, either "svd" (default) or "eigh".
    device : device or None, default=None
        Torch device.

    Returns
    -------
    Tensor
        Whitened signal with shape (n_components, n_samples).
    Tensor
        Estimated mean vector.
    Tensor
        Estimated whitening matrix.
    """


def pca_whitening(
    x: np.ndarray | torch.Tensor,
    n_pcs: int = -1,
    p_discard: float = 0,
    var_th: float = 1,
    solver: str = "svd",
    device: torch.device | None = None,
) -> tuple[np.ndarray | torch.Tensor, torch.Tensor, torch.Tensor]:
    """Function performing PCA whitening.

    Parameters
    ----------
    x : ndarray or Tensor
        Signal with shape (n_channels, n_samples).
    n_pcs : int, default=-1
        Number of components to be selected (if zero or negative, all components will be retained)).
    p_discard : float, default=0
        Proportion of components to be discarded; relevant if n_pcs is not specified.
    var_th : float, default=1
        Cut-off threshold for the variances; relevant if n_pcs and p_discard are not specified.
    solver : {"svd", "eigh"}, default="svd"
        The solver used for whitening, either "svd" (default) or "eigh".
    device : device or None, default=None
        Torch device.

    Returns
    -------
    ndarray or Tensor
        Whitened signal with shape (n_components, n_samples).
    Tensor
        Estimated mean vector.
    Tensor
        Estimated whitening matrix.
    """
    whiten_model = PCAWhitening(n_pcs, p_discard, var_th, solver, device)
    x_w = whiten_model.fit_transform(x)

    return x_w, whiten_model.mean_vec, whiten_model.white_mtx


class PCAWhitening(WhiteningModel):
    """Class implementing PCA whitening.

    Parameters
    ----------
    n_pcs : int, default=-1
        Number of components to be selected (if zero or negative, all components will be retained)).
    p_discard : float, default=0
        Proportion of components to be discarded; relevant if n_pcs is not specified.
    var_th : float, default=1
        Cut-off threshold for the variances; relevant if n_pcs and p_discard are not specified.
    solver : {"svd", "eigh"}, default="svd"
        The solver used for whitening, either "svd" (default) or "eigh".
    device : device or None, default=None
        Torch device.

    Attributes
    ----------
    _p_discard : float or None
        Proportion of components to be discarded.
    _var_th : float or None
        Cut-off threshold for the variances.
    _solver : str
        The solver used for whitening, either "svd" (default) or "eigh".
    _device : device or None
        Torch device.
    """

    def __init__(
        self,
        n_pcs: int = -1,
        p_discard: float = 0,
        var_th: float = 1,
        solver: str = "svd",
        device: torch.device | None = None,
    ) -> None:
        assert (
            p_discard is None or 0.0 <= p_discard < 1.0
        ), "The proportion of components to discard must be in [0, 1[ range."
        assert (
            var_th is None or 0.0 < var_th <= 1.0
        ), "The cut-off threshold must be in ]0, 1] range."
        assert solver in ("svd", "eigh"), 'The solver must be either "svd" or "eigh".'

        logging.info(f'Instantiating PCAWhitening using "{solver}" solver.')

        self._n_pcs: int = n_pcs
        self._p_discard: float = p_discard
        self._var_th: float = var_th
        self._solver: str = solver
        self._device: torch.device | None = device

        self._exp_var_ratio: np.ndarray | None = None
        self._mean_vec: torch.Tensor | None = None
        self._white_mtx: torch.Tensor | None = None

    @property
    def mean_vec(self) -> torch.Tensor | None:
        """Tensor: Property for getting the estimated mean vector."""
        return self._mean_vec

    @property
    def white_mtx(self) -> torch.Tensor | None:
        """Tensor: Property for getting the estimated whitening matrix."""
        return self._white_mtx

    @property
    def n_pcs(self) -> int:
        """int: Property for getting the number of principal components."""
        return self._n_pcs

    @property
    def exp_var_ratio(self) -> np.ndarray | None:
        """ndarray or None: Property for getting the estimated vector of explained variance ratio."""
        return self._exp_var_ratio

    def fit(self, x: np.ndarray | torch.Tensor) -> WhiteningModel:
        """Fit the whitening model on the given signal.

        Parameters
        ----------
        x : ndarray or Tensor
            Signal with shape (n_channels, n_samples).

        Returns
        -------
        WhiteningModel
            The fitted whitening model.
        """
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
            Whitened signal with shape (n_components, n_samples).
        """
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
            Whitened signal with shape (n_components, n_samples).
        """
        assert (
            self._mean_vec is not None and self._white_mtx is not None
        ), "Mean vector or whitening matrix are null, fit the model first."

        is_numpy = isinstance(x, np.ndarray)

        x_t = torch.from_numpy(x).to(self._device) if is_numpy else x.to(self._device)
        x_t -= self._mean_vec
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
        n_ch, n_samp = x_t.size()
        self._mean_vec = x_t.mean(dim=1, keepdim=True)
        x_t -= self._mean_vec

        if self._solver == "svd":
            e, d, _ = torch.linalg.svd(x_t, full_matrices=False)

            d_sq = d**2  # singular values are the square root of eigenvalues
            exp_var_ratio = (d_sq / d_sq.sum()).cpu().numpy()

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

            exp_var_ratio = (d / d.sum()).cpu().numpy()

            d_mtx = torch.diag(1.0 / torch.sqrt(d))
        else:
            raise NotImplementedError("Unknown solver.")

        e *= torch.sign(e[0])  # guarantee consistent sign

        # Select number of components to retain
        if self._n_pcs <= 0:
            if self._p_discard == 0:
                if self._var_th == 1:
                    # n_pcs, p_discard and var_th are not specified -> all components are retained
                    self._n_pcs = n_ch
                else:
                    # var_th is not specified -> choose components via explained variance
                    self._n_pcs = (
                        np.argmax(np.cumsum(exp_var_ratio) >= self._var_th).item() + 1
                    )
            else:
                # p_discard is not specified -> choose components using quantiles
                self._n_pcs = int(
                    torch.count_nonzero(
                        torch.ge(d, torch.quantile(d, self._p_discard))
                    ).item()
                )

        logging.info(f"Reducing dimension of data from {n_ch} to {self._n_pcs}.")
        d_mtx = d_mtx[: self._n_pcs, : self._n_pcs]
        e = e[:, : self._n_pcs]
        self._exp_var_ratio = exp_var_ratio[: self._n_pcs]

        self._white_mtx = d_mtx @ e.T
        x_w_t = self._white_mtx @ x_t

        if is_numpy:
            return x_w_t.cpu().numpy()
        else:
            return x_w_t
