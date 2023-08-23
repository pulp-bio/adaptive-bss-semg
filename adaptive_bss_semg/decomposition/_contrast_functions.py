"""Module containing contrast functions used in ICA algorithms.

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

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass
class ContrastFunctionResult:
    """Class representing the result of a contrast function.

    Attributes
    ----------
    g_u : Tensor
        The result itself.
    g1_u : Tensor
        The first derivative.
    g2_u : Tensor
        The second derivative.
    g_nu : float
        The value of the contrast function when evaluated on a standard Normal distribution.
    """

    g_u: torch.Tensor
    g1_u: torch.Tensor
    g2_u: torch.Tensor
    g_nu: float


ContrastFunction = Callable[[torch.Tensor], ContrastFunctionResult]


def logcosh(u: torch.Tensor) -> ContrastFunctionResult:
    """Function implementing the 'logcosh' nonlinearity.

    Parameters
    ----------
    u : Tensor
        Input array.

    Returns
    -------
    ContrastFunctionResult
        Object encompassing the result of the contrast function.
    """
    g_u = torch.log(torch.cosh(u))  # G(u) = logcosh(u)
    g1_u = torch.tanh(u)  # G'(u) = tanh(u)
    g2_u = 1 - g1_u**2  # G"(u) = 1 - tanh^2(u)

    return ContrastFunctionResult(g_u, g1_u, g2_u, 0.37456)


def gauss(u: torch.Tensor) -> ContrastFunctionResult:
    """Function implementing the 'gauss' nonlinearity.

    Parameters
    ----------
    u : Tensor
        Input array.

    Returns
    -------
    ContrastFunctionResult
        Object encompassing the result of the contrast function.
    """
    u_sq = u**2
    g_u = -torch.exp(-u_sq / 2)  # G(u) = -e^(-u^2 / 2)
    g1_u = -u * g_u  # G'(u) = u e^(-u^2 / 2)
    g2_u = (u_sq - 1) * g_u  # G"(u) = (1 - u^2) e^(-u^2 / 2)

    return ContrastFunctionResult(g_u, g1_u, g2_u, -0.70711)


def kurt(u: torch.Tensor) -> ContrastFunctionResult:
    """Function implementing the 'kurt' nonlinearity.

    Parameters
    ----------
    u : Tensor
        Input array.

    Returns
    -------
    ContrastFunctionResult
        Object encompassing the result of the contrast function.
    """
    u_sq = u**2
    g_u = u_sq**2 / 4  # G(u) = u^4 / 4
    g1_u = u_sq * u  # G'(u) = u^3
    g2_u = 3 * u_sq  # G"(u) = 3 u^2

    return ContrastFunctionResult(g_u, g1_u, g2_u, 3.0000)


def skew(u: torch.Tensor) -> ContrastFunctionResult:
    """Function implementing the 'skew' nonlinearity.

    Parameters
    ----------
    u : Tensor
        Input array.

    Returns
    -------
    ContrastFunctionResult
        Object encompassing the result of the contrast function.
    """
    u_sq = u**2
    g_u = u * u_sq / 3  # G(u) = u^3 / 3
    g1_u = u_sq  # G'(u) = u^2
    g2_u = 2 * u  # G"(u) = 2 u

    return ContrastFunctionResult(g_u, g1_u, g2_u, 0.015871)


def rati(u: torch.Tensor) -> ContrastFunctionResult:
    """Function implementing the 'rati' nonlinearity.

    Parameters
    ----------
    u : Tensor
        Input array.

    Returns
    -------
    ContrastFunctionResult
        Object encompassing the result of the contrast function.
    """
    u_sq = u**2
    g_u = torch.log(u_sq + 1) / 2  # G(u) = log(u^2 + 1) / 2
    g1_u = u / (1 + u_sq)  # G'(u) = u / (1 + u^2)
    g2_u = (1 - u_sq) / (1 + u_sq) ** 2  # G"(u) = (1 - u^2) / (1 + u^2)^2

    return ContrastFunctionResult(g_u, g1_u, g2_u, 3.1601)
