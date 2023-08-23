"""Utility functions for EMG decomposition.


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
from scipy import signal
from scipy.cluster.vq import kmeans2
from sklearn.metrics import silhouette_score


def detect_spikes(
    ic: np.ndarray | torch.Tensor,
    ref_period: int,
    threshold: float | None = None,
    compute_sil: bool = False,
    seed: int | np.random.Generator | None = None,
) -> tuple[np.ndarray, float, float]:
    """Detect spikes in the given IC.

    Parameters
    ----------
    ic : ndarray or Tensor
        Estimated IC with shape (n_samples,).
    ref_period : int
        Refractory period for spike detection.
    threshold : float or None, default=None
        Threshold for spike/noise classification.
    compute_sil : bool, default=False
        Whether to compute SIL measure or not.
    seed : int or Generator or None, default=None
        Seed for PRNG.

    Returns
    -------
    ndarray
        Location of spikes.
    float
        Threshold for spike/noise classification.
    float
        SIL measure.
    """
    ic_arr = ic if isinstance(ic, np.ndarray) else ic.cpu().numpy()

    peaks, _ = signal.find_peaks(ic_arr, height=0, distance=ref_period)
    ic_peaks = ic_arr[peaks]

    if threshold is None:
        centroids, labels = kmeans2(ic_peaks.reshape(-1, 1), k=2, minit="++", seed=seed)
        high_cluster_idx = np.argmax(centroids)  # consider only high peaks
        spike_loc = peaks[labels == high_cluster_idx]
        threshold = centroids.mean()
    else:
        labels = ic_peaks >= threshold
        spike_loc = peaks[labels]

    sil = np.nan
    if compute_sil:
        sil = float(silhouette_score(ic_peaks.reshape(-1, 1), labels))

    return spike_loc, threshold, sil
