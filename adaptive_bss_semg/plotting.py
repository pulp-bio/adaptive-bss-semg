"""Functions for visualizations (e.g., heatmap representing the spike count of a group of MUs).


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

from collections import Counter
from collections.abc import Collection
from itertools import groupby

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import cm
from matplotlib import patches as mpl_patches
from matplotlib import pyplot as plt

# Set Seaborn default theme
sns.set_theme()


def plot_signal(
    s: np.ndarray | pd.DataFrame | pd.Series | torch.Tensor,
    fs: float = 1.0,
    labels: np.ndarray | pd.Series | None = None,
    title: str | None = None,
    x_label: str = "Time [s]",
    y_label: str = "Amplitude [a.u.]",
    fig_size: tuple[int, int] | None = None,
    file_name: str | None = None,
) -> None:
    """Helper function to plot a signal with multiple channels, each in a different subplot.

    Parameters
    ----------
    s : ndarray or DataFrame or Series or Tensor
        Signal to plot:
        - if it's a NumPy array or PyTorch Tensor, the shape must be (n_channels, n_samples);
        - if it's a DataFrame or Series, the index and column(s) must represent
          the samples and the channel(s), respectively.
    fs : float, default=1.0
        Sampling frequency of the signal (relevant if s is a NumPy array).
    labels : ndarray or Series or None, default=None
        NumPy array or Series containing a label for each sample.
    title : str or None, default=None
        Title of the plot.
    x_label : str, default="Time [s]"
        Label for X axis.
    y_label : str, default="Amplitude [a.u.]"
        Label for Y axis.
    fig_size : tuple of (int, int) or None, default=None
        Height and width of the plot.
    file_name : str or None, default=None
        Name of the file where the image will be saved to.
    """
    # Convert signal to DataFrame
    if isinstance(s, pd.DataFrame):
        s_df = s
    elif isinstance(s, pd.Series):
        s_df = s.to_frame()
    else:
        s_arr = s.cpu().numpy() if isinstance(s, torch.Tensor) else s
        if len(s_arr.shape) == 1:
            s_arr = s_arr.reshape(1, -1)
        s_df = pd.DataFrame(s_arr.T, index=np.arange(s_arr.shape[1]) / fs)

    # Create figure with subplots and shared X axis
    n_cols = 1
    n_rows = s_df.shape[1]
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        sharex="all",
        squeeze=False,
        figsize=fig_size,
        layout="constrained",
    )
    axes = [ax for nested_ax in axes for ax in nested_ax]  # flatten axes
    # Set title and label of X and Y axes
    if title is not None:
        fig.suptitle(title, fontsize="xx-large")
    fig.supxlabel(x_label)
    fig.supylabel(y_label)

    # Plot signal
    if labels is not None:
        # Get label intervals
        labels_intervals = []
        labels_tmp = [
            list(group)
            for _, group in groupby(
                labels.reset_index().to_numpy().tolist(), key=lambda t: t[1]
            )
        ]
        for cur_label in labels_tmp:
            cur_label_start, cur_label_name = cur_label[0]
            cur_label_stop = cur_label[-1][0]
            labels_intervals.append((cur_label_name, cur_label_start, cur_label_stop))
        # Get set of unique labels
        label_set = set(map(lambda t: t[0], labels_intervals))
        # Create dictionary label -> color
        cmap = cm.get_cmap("plasma", len(label_set))
        color_dict = {lab: cmap(i) for i, lab in enumerate(label_set)}
        for i, ch_i in enumerate(s_df):
            for label, idx_from, idx_to in labels_intervals:
                axes[i].plot(
                    s_df[ch_i].loc[idx_from:idx_to],
                    color=color_dict[label],
                )
        # Add legend
        fig.legend(
            handles=[
                mpl_patches.Patch(color=c, label=lab) for lab, c in color_dict.items()
            ],
            loc="center right",
        )
    else:
        for i, ch_i in enumerate(s_df):
            axes[i].plot(s_df[ch_i])

    # Show or save plot
    if file_name is not None:
        plt.savefig(file_name)
    else:
        plt.show()


def plot_waveforms(
    wfs: np.ndarray,
    fs: float = 1.0,
    n_cols: int = 10,
    y_label: str = "Amplitude [a.u.]",
    fig_size: tuple[int, int] | None = None,
    file_name: str | None = None,
) -> None:
    """Function to plot MUAP waveforms.

    Parameters
    ----------
    wfs : ndarray
        MUAP waveforms with shape (n_mu, n_channels, waveform_len).
    fs : float, default=1.0
        Sampling frequency of the signal.
    n_cols : int, default=10
        Number of columns for subplots.
    y_label : str, default="Amplitude [a.u.]"
        Label for Y axis.
    fig_size : tuple of (int, int) or None, default=None
        Height and width of the plot.
    file_name : str or None, default=None
        Name of the file where the image will be saved to.
    """
    n_ch = wfs.shape[1]
    assert (
        n_ch % n_cols == 0
    ), "The number of channels must be divisible for the number of columns."
    n_rows = n_ch // n_cols
    t = np.arange(wfs.shape[2]) * 1000 / fs

    f, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        sharex="all",
        sharey="all",
        figsize=fig_size,
        layout="constrained",
    )

    for i in range(n_rows):
        for j in range(n_cols):
            idx = i * n_cols + j
            axes[i, j].set_title(f"Ch{idx}")
            axes[i, j].plot(t, wfs[:, idx].T)
            axes[i, j].axvline(t[wfs.shape[2] // 2], color="k", linestyle="--")

    f.suptitle("MUAP waveforms")
    f.supxlabel("Time [ms]")
    f.supylabel(y_label)

    if file_name is not None:
        plt.savefig(file_name)
    else:
        plt.show()


def plot_correlation(
    s: np.ndarray | pd.DataFrame | torch.Tensor,
    write_annotations: bool = False,
    fig_size: tuple[int, int] | None = None,
    file_name: str | None = None,
) -> None:
    """Plot the correlation matrix between the channels of a given signal.

    Parameters
    ----------
    s : ndarray or DataFrame or Tensor
        Signal:
        - if it's a NumPy array or PyTorch Tensor, the shape must be (n_channels, n_samples);
        - if it's a DataFrame, the index and columns must represent
          the samples and the channels, respectively.
    write_annotations : bool, default=False
        Whether to write annotations inside the correlation matrix or not.
    fig_size : tuple of (int, int) or None, default=None
        Height and width of the plot.
    file_name : str or None, default=None
        Name of the file where the image will be saved to.
    """
    # Convert to DataFrame
    if isinstance(s, pd.DataFrame):
        s_df = s
    else:
        s_arr = s.cpu().numpy() if isinstance(s, torch.Tensor) else s
        s_df = pd.DataFrame(s_arr.T)

    # Compute correlation and plot heatmap
    corr = s_df.corr()
    _, ax = plt.subplots(figsize=fig_size, layout="constrained")
    sns.heatmap(
        corr,
        vmax=1.0,
        vmin=-1.0,
        cmap="icefire",
        annot=write_annotations,
        square=True,
        ax=ax,
    )

    if file_name is not None:
        plt.savefig(file_name)
    else:
        plt.show()


def _data_distribution_helper(
    y: np.ndarray, label_dict: dict[str, int], ax: plt.Axes
) -> None:
    """Helper function to plot the distribution of the labels of a single dataset.

    Parameters
    ----------
    y : ndarray
        An array containing the labels of a dataset.
    label_dict : dict of {str, int}
        Dictionary mapping string labels to the respective integer labels.
    ax : Axes or None, default=None
        Matplotlib Axes object on which the bar plot is drawn.
    """
    # Count labels
    y_count = Counter(y)

    g_range = list(label_dict.values())
    ax.bar(
        x=g_range,
        height=[y_count[g] for g in g_range],
    )
    ax.set_xticks(g_range)
    ax.set_xticklabels(label_dict.keys(), rotation=45)


def data_distribution(
    y: np.ndarray | Collection[np.ndarray],
    label_dict: dict[str, int],
    title: str | Collection[str] | None = None,
    fig_size: tuple[int, int] | None = None,
    file_name: str | None = None,
) -> None:
    """Plot the distribution of the labels of a given dataset (or list of datasets).

    Parameters
    ----------
    y : ndarray or list of ndarray
        An array (or a list of arrays) containing the labels of a dataset.
    label_dict : dict of {str, int}
        Dictionary mapping string labels to the respective integer labels.
    title : str or list of str or None, default=None
        String (or list of strings) representing the title(s) of the plot(s).
    fig_size : tuple of (int, int) or None, default=None
        Height and width of the plot.
    file_name : str or None, default=None
        Name of the file where the image will be saved to.
    """
    # Check for single or multiple plots
    if isinstance(y, np.ndarray):  # single plot
        assert (
            isinstance(title, str) or title is None
        ), "'y' is a single array, thus 'title' should be single as well."

        # Create figure
        _, ax = plt.subplots(figsize=fig_size, layout="constrained")

        # Plot bar plot
        _data_distribution_helper(y, label_dict, ax)

        # Set title
        if title is not None:
            ax.set_title(title)
    elif isinstance(y, Collection):  # multiple plots
        assert (
            isinstance(title, Collection) and len(y) == len(title)
        ) or title is None, (
            "'y' is a list of arrays, thus 'title' (if provided) should be a list as well, "
            "with the same length as 'y'."
        )

        # Compute number of rows
        n_plots = len(y)
        n_cols = 2
        mod = n_plots % n_cols
        n_rows = n_plots // n_cols if mod == 0 else n_plots // n_cols + mod
        # Create figure with subplots and shared x-axis
        _, axes = plt.subplots(
            n_rows,
            n_cols,
            sharex="all",
            sharey="all",
            squeeze=False,
            figsize=fig_size,
            layout="constrained",
        )
        axes = [ax for nested_ax in axes for ax in nested_ax]  # flatten axes

        # Plot barplots
        opt_title_list = [None] * len(y) if title is None else title
        for y, t, a in zip(y, opt_title_list, axes):
            _data_distribution_helper(y, label_dict, a)
            # Set title
            if t is not None:
                a.set_title(t)
    else:
        raise NotImplementedError(
            "This function does not support the given parameters."
        )

    if file_name is not None:
        plt.savefig(file_name)
    else:
        plt.show()
