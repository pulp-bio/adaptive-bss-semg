"""Function and class implementing the ORICA algorithm
(https://doi.org/10.1109/TNSRE.2015.2508759).


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

import numpy as np
import torch


class ORICA:
    """Class implementing ORICA.

    Parameters
    ----------
    white_init : ndarray or Tensor
        Initial whitening matrix with shape (n_channels, n_channels).
    sep_init : ndarray or Tensor
        Initial separation matrix with shape (n_components, n_channels).
    lambda_0 : float, default=0.9
        Initial forgetting factor.
    gamma : float, default=0.6
        Decaying factor.
    device : device or None, default=None
        Torch device.

    Attributes
    ----------
    _lambda_0 : float
        Forgetting factor.
    _device : device or None
        Torch device.
    _n : int
        Iteration count.
    """

    def __init__(
        self,
        white_init: np.ndarray | torch.Tensor,
        sep_init: np.ndarray | torch.Tensor,
        lambda_0: float = 0.9,
        gamma: float = 0.6,
        device: torch.device | None = None,
    ) -> None:
        assert (
            0 < lambda_0 < 1
        ), f"The forgetting factor must be in ]0, 1[ range, the one provided was {lambda_0}."

        self._lambda_0 = lambda_0
        self._gamma = gamma
        self._device = device

        self._white_mtx: torch.Tensor = (
            torch.from_numpy(white_init).to(device)
            if isinstance(white_init, np.ndarray)
            else white_init
        )
        self._sep_mtx: torch.Tensor = (
            torch.from_numpy(sep_init).to(device)
            if isinstance(sep_init, np.ndarray)
            else sep_init
        )
        self._n: int = 1

    @property
    def white_mtx(self) -> torch.Tensor:
        """Tensor: Property for getting the estimated whitening matrix."""
        return self._white_mtx

    @property
    def sep_mtx(self) -> torch.Tensor:
        """Tensor: Property for getting the estimated separation matrix."""
        return self._sep_mtx

    def fit_transform(
        self, x: np.ndarray | torch.Tensor, approx: bool = False
    ) -> np.ndarray | torch.Tensor:
        """Decompose the given signal.

        Parameters
        ----------
        x : ndarray or Tensor
            Signal with shape (n_channels, n_samples).
        approx : bool, default=True
            Whether to use the approximate orthogonalization.

        Returns
        -------
        ndarray or Tensor
            Estimated source signal with shape (n_components, n_samples).
        """

        # Contrast function for super-gaussian sources
        def g_fn(x_):
            return -2 * torch.tanh(x_)

        def sym_orth(w_: torch.Tensor) -> torch.Tensor:
            eig_vals, eig_vecs = torch.linalg.eigh(w_ @ w_.T)

            # Improve numerical stability
            eig_vals = torch.clip(eig_vals, min=torch.finfo(w_.dtype).tiny)

            d_mtx = torch.diag(1.0 / torch.sqrt(eig_vals))
            return eig_vecs @ d_mtx @ eig_vecs.T @ w_

        def sym_orth_approx(w_: torch.Tensor) -> torch.Tensor:
            max_iter = 8

            for _ in range(max_iter):
                w_ /= torch.linalg.norm(w_, ord=1).item()
                w_ = 3 / 2 * w_ - 1 / 2 * w_ @ w_.T @ w_

            return w_

        is_numpy = isinstance(x, np.ndarray)
        x_t = torch.from_numpy(x).to(self._device) if is_numpy else x.to(self._device)
        n_samp = x_t.size(dim=1)
        lambda_n = self._lambda_0 / self._n**self._gamma
        if lambda_n < 1e-4:
            lambda_n = 1e-4
        beta = (1 - lambda_n) / lambda_n

        # Whitening
        white_mtx_old = self._white_mtx
        v = white_mtx_old @ x_t
        cov_mtx = v @ v.T / n_samp
        norm_factor = beta + (v * v).sum().item() / n_samp
        self._white_mtx = (
            1 / (1 - lambda_n) * (white_mtx_old - cov_mtx / norm_factor @ white_mtx_old)
        )
        v = self._white_mtx @ x_t
        w_diff = torch.abs(self._white_mtx - white_mtx_old).mean().item()

        # Separation
        sep_mtx_old = self._sep_mtx
        y = sep_mtx_old @ v
        g = g_fn(y)
        cov_mtx = y @ g.T / n_samp
        norm_factor = beta + (g * y).sum().item() / n_samp
        self._sep_mtx = (
            1 / (1 - lambda_n) * (sep_mtx_old - cov_mtx / norm_factor @ sep_mtx_old)
        )
        self._sep_mtx = (
            sym_orth_approx(self._sep_mtx) if approx else sym_orth(self._sep_mtx)
        )
        y = self._sep_mtx @ v
        s_diff = torch.abs(self._sep_mtx - sep_mtx_old).mean().item()
        self._n += 1  # n_samp

        if is_numpy:
            return v.cpu().numpy(), y.cpu().numpy(), lambda_n, w_diff, s_diff
        else:
            return v, y, lambda_n, w_diff, s_diff
