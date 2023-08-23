"""Function performing the extension step of an EMG signal.


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


def extend_signal(x: np.ndarray, f_ext: int = 1) -> np.ndarray:
    """Extend signal with delayed replicas by a given extension factor.

    Parameters
    ----------
    x : ndarray
        Signal with shape:
        - (n_samples,);
        - (n_channels, n_samples).
    f_ext : int, default=1
        Extension factor.

    Returns
    -------
    ndarray
        Extended signal with shape (f_ext * n_channels, n_samples).
    """
    if len(x.shape) == 1:
        x = x.reshape(1, -1)

    n_ch, n_samp = x.shape
    n_ch_ext = f_ext * n_ch

    x_ext = np.zeros(shape=(n_ch_ext, n_samp), dtype=x.dtype)
    for i in range(f_ext):
        x_ext[i * n_ch : (i + 1) * n_ch, i:] = x[:, : n_samp - i]
    return x_ext
