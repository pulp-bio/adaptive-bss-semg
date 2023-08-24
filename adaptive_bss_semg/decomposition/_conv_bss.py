"""Class implementing the convolutive blind source separation algorithm
for EMG decomposition (https://doi.org/10.1088/1741-2560/13/2/026027).


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

import heapq
import logging
import math
import time

import numpy as np
import pandas as pd
import torch
from scipy.signal import find_peaks

from adaptive_bss_semg import preprocessing, spike_stats, utils

from . import _contrast_functions as cf


class ConvBSS:
    """Decompose EMG signals via convolutive blind source separation.

    Parameters
    ----------
    fs : float
        Sampling frequency of the signal.
    f_ext_ms : float, default=-1
        Extension factor for the signal (in ms):
        - if zero, the signal won't be extended;
        - if negative, it will be set to 1000 / n. of channels.
    n_mu_target : int, default=-1
        Number of target MUs to extract (if zero or negative, it will be set to the number of extended observations).
    ref_period_ms : float, default=20.0
        Refractory period for spike detection (in ms).
    g_name : {"logcosh", "gauss", "kurt", "skew", "rati"}, default="logcosh"
        Name of the contrast function.
    conv_th : float, default=1e-4
        Threshold for convergence.
    max_iter : int, default=200
        Maximum n. of iterations.
    sil_th : float, default=0.85
        Minimum silhouette threshold for considering a MU as valid.
    fr_th : float, default=10.0
        Minimum firing rate (in pulses per second) for considering a MU as valid.
    dup_perc : float, default=0.5
        Minimum percentage of common firings for considering two MUs as duplicates.
    dup_tol_ms : float, default=0.5
        Tolerance (in ms) of firings alignment for considering two MUs as duplicates.
    exclude_long_intervals : bool, default=False
        Whether to exclude intervals > 250ms (i.e., corresponding to rest periods) from the CoV-ISI computation.
    seed : int or None, default=None
        Seed for the internal PRNG.
    device : device or None, default=None
        Torch device.

    Attributes
    ----------
    _fs : float
        Sampling frequency of the signal.
    _f_ext : int
        Extension factor for the signal.
    _n_mu_target : int
        Number of target MUs to extract.
    _ref_period : float
        Refractory period for spike detection.
    _g_func : ContrastFunction
        Contrast function.
    _conv_th : float
        Threshold for convergence.
    _max_iter : int
        Maximum n. of iterations.
    _sil_th : float
        Minimum silhouette threshold for considering a MU as valid.
    _fr_th : float
        Minimum firing rate (in pulses per second) for considering a MU as valid.
    _dup_perc : float
        Minimum percentage of common firings for considering two MUs as duplicates.
    _dup_tol_ms : float
        Tolerance (in ms) of firings alignment for considering two MUs as duplicates.
    _exclude_long_intervals : bool
        Whether to exclude intervals > 250ms (i.e., corresponding to rest periods) from the CoV-ISI computation.
    _prng : Generator
        Actual PRNG.
    _device : device or None
        Torch device.
    """

    def __init__(
        self,
        fs: float,
        f_ext_ms: float = -1,
        n_mu_target: int = -1,
        ref_period_ms: float = 20.0,
        g_name: str = "logcosh",
        conv_th: float = 1e-4,
        max_iter: int = 100,
        sil_th: float = 0.85,
        fr_th: float = 10.0,
        dup_perc: float = 0.5,
        dup_tol_ms: float = 0.5,
        exclude_long_intervals: bool = False,
        seed: int | None = None,
        device: torch.device | None = None,
    ):
        assert g_name in (
            "logcosh",
            "gauss",
            "kurt",
            "skew",
            "rati",
        ), (
            'Contrast function can be either "logcosh", "gauss", "kurt", "skew" or "rati": '
            f'the provided one was "{g_name}".'
        )
        assert conv_th > 0, "Convergence threshold must be positive."
        assert max_iter > 0, "The maximum n. of iterations must be positive."

        self._fs: float = fs
        self._n_mu_target: int = n_mu_target
        self._ref_period: int = int(round(ref_period_ms / 1000 * fs))
        g_dict = {
            "logcosh": cf.logcosh,
            "gauss": cf.gauss,
            "kurt": cf.kurt,
            "skew": cf.skew,
            "rati": cf.rati,
        }
        self._g_func: cf.ContrastFunction = g_dict[g_name]
        self._conv_th: float = conv_th
        self._max_iter: int = max_iter
        self._sil_th: float = sil_th
        self._fr_th: float = fr_th
        self._dup_perc: float = dup_perc
        self._dup_tol_ms: float = dup_tol_ms
        self._exclude_long_intervals: bool = exclude_long_intervals
        self._prng: np.random.Generator = np.random.default_rng(seed)
        self._device: torch.device | None = device

        if f_ext_ms == 0:  # disable extension
            self._f_ext: int = 1
        elif f_ext_ms < 0:  # apply heuristic later
            self._f_ext: int = int(f_ext_ms)
        else:  # convert from ms to samples
            self._f_ext: int = int(round(f_ext_ms / 1000 * fs))

        if seed is not None:
            torch.manual_seed(seed)

        self._mean_vec: torch.Tensor | None = None
        self._white_mtx: torch.Tensor | None = None
        self._sep_mtx: torch.Tensor | None = None
        self._spike_ths: np.ndarray | None = None

    @property
    def mean_vec(self) -> torch.Tensor | None:
        """Tensor or None: Property for getting the estimated mean vector."""
        return self._mean_vec

    @property
    def white_mtx(self) -> torch.Tensor | None:
        """Tensor or None: Property for getting the estimated whitening matrix."""
        return self._white_mtx

    @property
    def sep_mtx(self) -> torch.Tensor | None:
        """Tensor or None: Property for getting the estimated separation matrix."""
        return self._sep_mtx

    @property
    def spike_ths(self) -> np.ndarray | None:
        """ndarray or None: Property for getting the estimated separation matrix."""
        return self._spike_ths

    @property
    def n_mu(self) -> int:
        """int: Number of identified motor units."""
        return self._sep_mtx.size(dim=0) if self._sep_mtx is not None else 0

    @property
    def f_ext(self) -> int:
        """int: Property representing the extension factor."""
        return self._f_ext

    def fit(self, emg: np.ndarray) -> ConvBSS:
        """Fit the decomposition model on the given data.

        Parameters
        ----------
        emg : ndarray
            EMG signal with shape (n_channels, n_samples).

        Returns
        -------
        ConvBSS
            The fitted instance of the decomposition model.
        """
        # Fit the model and return self
        self._fit_transform(emg)
        return self

    def fit_transform(self, emg: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Fit the decomposition model on the given data and compute the estimated MUAPTs.

        Parameters
        ----------
        emg : ndarray
            EMG signal with shape (n_channels, n_samples).

        Returns
        -------
        DataFrame
            A DataFrame with one column "MU{i}" for each MU and one row for each sample in the signal,
            containing the source before binarization.
        DataFrame
            A DataFrame with one column "MU{i}" for each MU and one row for each sample in the signal,
            containing the source after binarization (i.e., either ones/zeros for spike/not spike).
        """
        # Fit the model and return result
        return self._fit_transform(emg)

    def transform(self, emg: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Compute the estimated MUAPTs using the fitted decomposition model.

        Parameters
        ----------
        emg : ndarray
            EMG signal with shape (n_channels, n_samples).

        Returns
        -------
        DataFrame
            A DataFrame with one column "MU{i}" for each MU and one row for each sample in the signal,
            containing the source before binarization.
        DataFrame
            A DataFrame with one column "MU{i}" for each MU and one row for each sample in the signal,
            containing the source after binarization (i.e., either ones/zeros for spike/not spike).
        """
        assert (
            self._mean_vec is not None
            and self._sep_mtx is not None
            and self._spike_ths is not None
        ), "Mean vector, separation matrix or spike-noise thresholds are null, fit the model first."

        # 1. Extension
        emg_ext = preprocessing.extend_signal(emg, self._f_ext)
        n_samp = emg_ext.shape[1]
        emg_ext_t = torch.from_numpy(emg_ext).to(self._device)

        # 2. Whitening + ICA
        ics_t = self._sep_mtx @ self._white_mtx @ (emg_ext_t - self._mean_vec)

        # 3. Binarization
        ics_bin = np.zeros(shape=ics_t.shape, dtype=np.uint8)
        n_mu = ics_bin.shape[0]
        for i in range(n_mu):
            spike_loc_i = utils.detect_spikes(
                ics_t[i],
                ref_period=self._ref_period,
                threshold=self._spike_ths[i].item(),
                seed=self._prng,
            )[0]
            ics_bin[i, spike_loc_i] = 1

        # Pack results in a DataFrame
        muapts = pd.DataFrame(
            data=ics_t.cpu().numpy().T,
            index=[i / self._fs for i in range(n_samp)],
            columns=[f"MU{i}" for i in range(n_mu)],
        )
        muapts_bin = pd.DataFrame(
            data=ics_bin.T,
            index=[i / self._fs for i in range(n_samp)],
            columns=[f"MU{i}" for i in range(n_mu)],
        )

        return muapts, muapts_bin

    def _fit_transform(self, emg: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Helper method for fit and fit_transform."""
        start = time.time()

        n_ch = emg.shape[0]

        # Apply heuristic
        if self._f_ext < 0:
            self._f_ext = int(round(1000 / n_ch))

        # 1. Extension
        logging.info(f"Number of channels before extension: {n_ch}")
        emg_ext = preprocessing.extend_signal(emg, self._f_ext)
        n_samp = emg_ext.shape[1]
        logging.info(f"Number of channels after extension: {emg_ext.shape[0]}")
        emg_ext_t = torch.from_numpy(emg_ext).to(self._device)

        # 2. Whitening
        emg_white_t: torch.Tensor
        emg_white_t, self._mean_vec, self._white_mtx = preprocessing.zca_whitening(
            emg_ext_t, device=self._device
        )

        if self._n_mu_target <= 0:
            self._n_mu_target = emg_white_t.size(dim=0)

        # 3. ICA
        self._sep_mtx = torch.zeros(
            0, emg_white_t.size(dim=0), dtype=emg_white_t.dtype, device=self._device
        )
        self._spike_ths = np.zeros(shape=0, dtype=emg.dtype)
        w_init = self._initialize_weights(emg, emg_white_t)
        ics_imp = np.zeros(shape=(0, n_samp), dtype=np.uint8)
        ics_imp_bin = np.zeros(shape=(0, n_samp), dtype=np.uint8)  # binarized ICs
        sil_scores = []
        idx = 0
        for i in range(self._n_mu_target):
            logging.info(f"----- IC {i + 1} -----")

            w_i_t, converged = self._fast_ica_iter(emg_white_t, w_i_init_t=w_init[i])
            if not converged:
                logging.info("FastICA didn't converge, reinitializing...")
                continue

            # Solve sign uncertainty
            ic_i_t: torch.Tensor = w_i_t @ emg_white_t
            if (ic_i_t**3).mean() < 0:
                w_i_t *= -1

            # Do IC improvement
            w_i_t, spike_loc_i, spike_th_i, sil = self._ic_improvement(
                emg_white_t, w_i_t=w_i_t
            )
            if w_i_t is None:
                logging.info("IC improvement iteration failed, skipping IC.")
                continue

            if sil <= self._sil_th:
                logging.info(f"SIL below threshold (SIL = {sil:.3f}), skipping IC.")
                continue

            avg_fr = spike_stats.avg_firing_rate(spike_loc_i / self._fs)

            if avg_fr <= self._fr_th:
                logging.info(
                    f"Firing rate below threshold (FR = {avg_fr:.3f}), skipping IC."
                )
                continue

            cov_isi = spike_stats.cov_isi(
                spike_loc_i / self._fs, self._exclude_long_intervals
            )

            if cov_isi >= 0.6:
                logging.info(
                    f"CoV-ISI above threshold (CoV-ISI = {cov_isi:.3f}), skipping IC."
                )
                continue

            # Save separation vector and spike/noise threshold
            self._sep_mtx = torch.vstack((self._sep_mtx, w_i_t))
            self._spike_ths = np.append(self._spike_ths, spike_th_i)
            logging.info(f"SIL = {sil:.3f}")
            logging.info(f"Avg. firing rate = {avg_fr:.3f}spike/s")
            logging.info(f"CoV-ISI = {cov_isi:.2%}")
            logging.info(f"-> MU accepted (n. of MUs: {self._sep_mtx.shape[0]}).")

            # Save current IC (both continuous and binary versions)
            ic_i_t = w_i_t @ emg_white_t
            ic_i_bin = np.zeros(shape=n_samp, dtype=np.uint8)
            ic_i_bin[spike_loc_i] = 1
            ics_imp = np.vstack((ics_imp, ic_i_t.cpu().numpy()))
            ics_imp_bin = np.vstack((ics_imp_bin, ic_i_bin))
            # Save SIL
            sil_scores.append((idx, sil))
            idx += 1

        logging.info(f"Extracted {ics_imp_bin.shape[0]} MUs before replicas removal.")

        # 5. Duplicates removal
        logging.info("Looking for delayed replicas...")
        duplicate_mus = utils.find_replicas(
            ics_imp_bin, fs=self._fs, tol_ms=self._dup_tol_ms, min_perc=self._dup_perc
        )
        mus_to_remove = []
        for main_mu, dup_mus in duplicate_mus.items():
            # Unify duplicate MUs
            dup_mus = [main_mu] + dup_mus
            dup_str = ", ".join([f"{mu}" for mu in dup_mus])
            logging.info(f"Found group of duplicate MUs: {dup_str}.")

            # Keep only the MU with the highest SIL
            sil_scores_dup = list(filter(lambda t: t[0] in dup_mus, sil_scores))
            mu_keep = max(sil_scores_dup, key=lambda t: t[1])
            logging.info(f"Keeping MU {mu_keep[0]} (SIL = {mu_keep[1]:.3f}).")

            # Mark duplicates
            dup_mus.remove(mu_keep[0])
            mus_to_remove.extend(dup_mus)
        # Remove duplicates
        self._sep_mtx = torch.from_numpy(
            np.delete(self._sep_mtx.cpu().numpy(), mus_to_remove, axis=0)
        ).to(self._device)
        self._spike_ths = np.delete(self._spike_ths, mus_to_remove, axis=0)
        ics_imp = np.delete(ics_imp, mus_to_remove, axis=0)
        ics_imp_bin = np.delete(ics_imp_bin, mus_to_remove, axis=0)

        logging.info(f"Extracted {ics_imp_bin.shape[0]} MUs after replicas removal.")

        # Pack results in a DataFrame
        n_mu = ics_imp_bin.shape[0]
        muapts = pd.DataFrame(
            data=ics_imp.T,
            index=[i / self._fs for i in range(n_samp)],
            columns=[f"MU{i}" for i in range(n_mu)],
        )
        muapts_bin = pd.DataFrame(
            data=ics_imp_bin.T,
            index=[i / self._fs for i in range(n_samp)],
            columns=[f"MU{i}" for i in range(n_mu)],
        )

        elapsed = int(round(time.time() - start))
        hours, rem = divmod(elapsed, 3600)
        mins, secs = divmod(rem, 60)
        logging.info(
            f"Decomposition performed in {hours:d}h {mins:02d}min {secs:02d}s."
        )

        return muapts, muapts_bin

    def _initialize_weights(
        self, emg: np.ndarray, emg_white_t: torch.Tensor
    ) -> torch.Tensor:
        """Initialize separation vectors."""
        if len(emg.shape) == 3:
            emg = np.concatenate(emg, axis=-1)

        gamma = (emg**2).sum(axis=0)  # activation index
        peaks, _ = find_peaks(gamma)
        # Tuple (peak index, peak value)
        peaks_tup = [(p, gamma[p]) for p in peaks]
        n_top_peaks = peaks.size // 2  # keep the 50% highest peaks
        top_peaks_tup = heapq.nlargest(n_top_peaks, peaks_tup, lambda t: t[1].item())
        top_peaks, _ = zip(*top_peaks_tup)  # keep only peak indices
        top_peaks = np.asarray(top_peaks, dtype=int)

        if top_peaks.size > self._n_mu_target:
            init_idx = self._prng.choice(
                top_peaks, size=self._n_mu_target, replace=False
            )
            return torch.linalg.pinv(emg_white_t[:, init_idx])

        w_init = torch.linalg.pinv(emg_white_t[:, top_peaks])
        n_init_rand = self._n_mu_target - top_peaks.size
        w_init_rand = torch.randn(
            n_init_rand,
            emg_white_t.size(dim=0),
            dtype=emg_white_t.dtype,
            device=self._device,
        )
        return torch.cat((w_init, w_init_rand))

    def _fast_ica_iter(
        self, x_w_t: torch.Tensor, w_i_init_t: torch.Tensor
    ) -> tuple[torch.Tensor, bool]:
        """FastICA iteration."""
        w_i = w_i_init_t
        w_i /= torch.linalg.norm(w_i)

        iter_idx = 1
        converged = False
        while iter_idx <= self._max_iter:
            g_res = self._g_func(w_i @ x_w_t)
            w_i_new = (x_w_t * g_res.g1_u).mean(dim=1) - g_res.g2_u.mean() * w_i
            w_i_new -= (
                w_i_new @ self._sep_mtx.T @ self._sep_mtx
            )  # Gram-Schmidt decorrelation
            w_i_new /= torch.linalg.norm(w_i_new)

            distance = 1 - abs((w_i_new @ w_i).item())
            logging.info(f"FastICA iteration {iter_idx}: {distance:.3e}.")

            w_i = w_i_new

            if distance < self._conv_th:
                converged = True
                logging.info(
                    f"FastICA converged after {iter_idx} iterations, the distance is: {distance:.3e}."
                )
                break

            iter_idx += 1

        return w_i, converged

    def _ic_improvement(
        self,
        emg_white_t: torch.Tensor,
        w_i_t: torch.Tensor,
    ) -> tuple[torch.Tensor | None, np.ndarray, float, float]:
        """IC improvement iteration."""

        ic_i_t = w_i_t @ emg_white_t
        spike_loc, spike_th, sil = utils.detect_spikes(
            ic_i_t,
            ref_period=self._ref_period,
            compute_sil=True,
            seed=self._prng,
        )
        cov_isi = spike_stats.cov_isi(
            spike_loc / self._fs, self._exclude_long_intervals
        )
        iter_idx = 0
        if math.isnan(cov_isi):
            logging.info("Spike detection failed, aborting.")
            return None, spike_loc, np.nan, np.nan

        while True:
            w_i_new_t = emg_white_t[:, spike_loc].mean(dim=1)
            w_i_new_t /= torch.linalg.norm(w_i_new_t)

            ic_i_t = w_i_new_t @ emg_white_t
            spike_loc_new, spike_th_new, sil_new = utils.detect_spikes(
                ic_i_t,
                ref_period=self._ref_period,
                compute_sil=True,
                seed=self._prng,
            )
            cov_isi_new = spike_stats.cov_isi(
                spike_loc_new / self._fs, self._exclude_long_intervals
            )
            iter_idx += 1

            if math.isnan(cov_isi_new):
                logging.info(
                    f"Spike detection failed after {iter_idx} steps, aborting."
                )
                break
            if cov_isi_new >= cov_isi:
                logging.info(
                    f"CoV-ISI increased from {cov_isi:.2%} to {cov_isi_new:.2%} "
                    f"after {iter_idx} steps, aborting."
                )
                break
            logging.info(
                f"CoV-ISI decreased from {cov_isi:.2%} to {cov_isi_new:.2%} "
                f"after {iter_idx} steps."
            )
            w_i_t = w_i_new_t
            cov_isi = cov_isi_new
            spike_loc = spike_loc_new
            spike_th = spike_th_new
            sil = sil_new

        return w_i_t, spike_loc, spike_th, sil
