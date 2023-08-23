"""Functions for computing statistics of spike trains.


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

import statistics

import numpy as np


def avg_firing_rate(
    spikes_t: np.ndarray,
    win_len: float = 0.2,
    step_len: float = 0.1,
) -> float:
    """Estimate the average firing rate of the given spike train.

    Parameters
    ----------
    spikes_t : ndarray
        Array containing the time of spikes (in seconds).
    win_len : float, default=0.2
        Length of the window (in seconds) to consider for the spike count computation.
    step_len : float, default=0.1
        Step size (in seconds) between consecutive windows for the spike count computation.
    """
    n_steps = 0
    avg_fr = 0.0
    start = np.min(spikes_t)
    stop = np.max(spikes_t)
    while True:
        # Compute firing rate in current window
        cur_fr = (
            np.count_nonzero((spikes_t >= start) & (spikes_t < start + win_len))
            / win_len
        )
        # Update average only if the neuron is active
        if cur_fr != 0:
            avg_fr += cur_fr
            n_steps += 1
        start += step_len

        if start + win_len >= stop:
            break
    if n_steps != 0:
        avg_fr /= n_steps
    return avg_fr


def cov_isi(spikes_t: np.ndarray, exclude_long_intervals: bool = True) -> float:
    """Compute the Coefficient of Variation of the Inter-Spike Interval (CoV-ISI) of the given spike train.

    Parameters
    ----------
    spikes_t : ndarray
        Array containing the time of spikes (in seconds).
    exclude_long_intervals : bool, default=True
        Whether to exclude intervals > 250ms (i.e., corresponding to rest periods) from the CoV-ISI computation.

    Returns
    -------
    float
        The CoV-ISI of the spike train.
    """
    # Compute ISI
    isi = np.diff(spikes_t).astype(float)
    if exclude_long_intervals:
        isi = isi[isi < 0.25]

    res = np.nan
    if isi.size > 1:
        # Compute CoV-ISI
        res = statistics.stdev(isi) / statistics.mean(isi)
    return res


def cov_amp(spikes_amp: np.ndarray) -> float:
    """Compute the Coefficient of Variation of the spike amplitude of the given spike train.

    Parameters
    ----------
    spikes_amp : ndarray
        Array containing the amplitude of spikes.

    Returns
    -------
    float
        The CoV-ISI of the spike train.
    """
    res = np.nan
    if spikes_amp.size > 1:
        # Compute CoV-Amp
        res = statistics.stdev(spikes_amp) / statistics.mean(spikes_amp)
    return res
