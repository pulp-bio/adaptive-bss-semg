"""Utility functions (e.g., to slice EMG signals by time or label, or to estimate the firing rate of a spike train).


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


def _compute_delay(s1: np.ndarray, s2: np.ndarray) -> int:
    """Find the lag between two pulse trains with the same length."""

    # Compute cross-correlation
    corr = signal.correlate(s2, s1, mode="same")
    delay_steps = int(round(s1.shape[0] / 2))
    delay_arr = np.arange(-delay_steps, delay_steps)

    # Return optimal delay
    return delay_arr[np.argmax(corr)].item()


def check_delayed_pair(
    ref_pulses: np.ndarray,
    sec_pulses: np.ndarray,
    fs: float,
    tol_ms: float,
    min_perc: float,
) -> tuple[bool, int, int, int, int]:
    """Check if two pulse trains are the same up to a delay by counting the common pulses.

    Parameters
    ----------
    ref_pulses : ndarray
        Reference pulse train represented as an array of 1s and 0s with shape (n_pulses1,).
    sec_pulses : ndarray
        Secondary pulse train represented as an array of 1s and 0s with shape (n_pulses2,).
    fs : float
        Sampling frequency of the pulse trains.
    tol_ms : float
        Tolerance for considering two pulses as synchronized.
    min_perc : float
        Minimum percentage of common pulses for considering the two pulse trains as the same.

    Returns
    -------
    bool
        Whether the two pulse trains are the same or not.
    int
        Number of samples representing the lag between the pulse trains.
    int
        Number of TPs if the pulse trains are the same, zero otherwise.
    int
        Number of FPs if the pulse trains are the same, zero otherwise.
    int
        Number of FNs if the pulse trains are the same, zero otherwise.
    """
    assert (
        ref_pulses.shape == sec_pulses.shape
    ), "The two pulse trains must have the same length."
    assert len(ref_pulses.shape) == 1, "The pulse trains must be 1D."

    # Find delay between reference and secondary pulse trains
    delay = _compute_delay(ref_pulses, sec_pulses)

    # Adjust for delay and get time of pulses
    ref_pulses_t = np.flatnonzero(ref_pulses) / fs
    sec_pulses_t = (np.flatnonzero(sec_pulses) - delay) / fs  # compensate for delay

    # Filter secondary pulses
    n_sec = sec_pulses_t.size
    sec_pulses_t = sec_pulses_t[sec_pulses_t >= 0]

    if ref_pulses_t.size == 0 or sec_pulses_t.size == 0:
        return False, delay, 0, 0, 0

    # Check pulse correspondence and count TP, FP and FN
    tol_s = tol_ms / 1000
    tp, fn = 0, 0
    for ref_pulse_t in ref_pulses_t:
        common_pulses = np.count_nonzero(
            (sec_pulses_t >= ref_pulse_t - tol_s)
            & (sec_pulses_t <= ref_pulse_t + tol_s)
        )
        if common_pulses == 0:  # no pulses found near the reference pulse -> one FN
            fn += 1
        elif common_pulses == 1:  # one pulse found near the reference pulse -> one TP
            tp += 1
    # The difference between the n. of secondary pulses and
    # the n. of TPs yields the n. of FPs
    fp = n_sec - tp

    # The pulse trains are the same if TPs > 30%
    same1 = tp / ref_pulses_t.size > min_perc
    same2 = tp / n_sec > min_perc

    return same1 and same2, delay, tp, fp, fn


def find_replicas(
    pulse_trains: np.ndarray,
    fs: float,
    tol_ms: float,
    min_perc: float,
) -> dict[int, list[int]]:
    """Given a set of pulse trains, find delayed replicas by checking each pair.

    Parameters
    ----------
    pulse_trains : ndarray
        Set of pulse trains represented as arrays of 1s and 0s with shape (n_trains, n_samples).
    fs : float
        Sampling frequency of the pulse trains.
    tol_ms : float
        Tolerance for considering two pulses as synchronized.
    min_perc : float
        Minimum percentage of common pulses for considering the two pulse trains as the same.

    Returns
    -------
    dict of (int: list of int)
        Dictionary containing delayed replicas.
    """
    n_trains = pulse_trains.shape[0]

    # Convert to dictionary
    pulse_train_dict = {i: pulse_trains[i] for i in range(n_trains)}

    # Check each pair
    cur_tr = 0
    tr_idx = list(pulse_train_dict.keys())
    duplicate_tr = {}
    while cur_tr < len(tr_idx):
        # Find index of replicas by checking synchronization
        i = 1
        while i < len(tr_idx) - cur_tr:
            # Find delay in binarized sources
            same = check_delayed_pair(
                ref_pulses=pulse_train_dict[tr_idx[cur_tr]],
                sec_pulses=pulse_train_dict[tr_idx[cur_tr + i]],
                fs=fs,
                tol_ms=tol_ms,
                min_perc=min_perc,
            )[0]

            if same:
                duplicate_tr[tr_idx[cur_tr]] = duplicate_tr.get(tr_idx[cur_tr], []) + [
                    tr_idx[cur_tr + i]
                ]
                del tr_idx[cur_tr + i]
            else:
                i += 1
        cur_tr += 1

    return duplicate_tr


def compute_waveforms(
    emg: np.ndarray,
    muapts_bin: np.ndarray,
    wf_len: int,
    device: torch.device | None = None,
) -> np.ndarray:
    """Compute the MUAPT waveforms.

    Parameters
    ----------
    emg : ndarray
        Raw EMG signal with shape (n_channels, n_samples).
    muapts_bin : ndarray
        Binary matrix representing the detected MUAPTs with shape (n_mu, n_samples).
    wf_len : int
        Length of the waveform.
    device : device or None, default=None
        Torch device.

    Returns
    -------
    ndarray
        MUAPT waveforms with shape (n_mu, n_channels, waveform_len).
    """
    n_ch, n_samp = emg.shape
    n_mu = muapts_bin.shape[0]
    wf_hlen = wf_len // 2

    emg_t = torch.from_numpy(emg).to(device)

    wfs = np.zeros(shape=(n_mu, n_ch, wf_len), dtype=emg.dtype)
    for i in range(n_mu):
        # Extend i-th MUAPT
        muapt_i_ext = np.zeros(shape=(wf_len, n_samp), dtype=emg.dtype)
        for j in range(wf_len):
            start = max(0, wf_hlen - j)
            stop = min(n_samp, n_samp + wf_hlen - j)
            start_ext = max(0, j - wf_hlen)
            stop_ext = min(n_samp, n_samp + j - wf_hlen)
            muapt_i_ext[j, start_ext:stop_ext] = muapts_bin[i, start:stop]

        # Apply Least Squares
        muapt_i_ext_t = torch.from_numpy(muapt_i_ext).to(device)
        wfs_i = torch.linalg.lstsq(muapt_i_ext_t.T, emg_t.T)[0]
        wfs[i] = wfs_i.cpu().numpy().T

    return wfs


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
