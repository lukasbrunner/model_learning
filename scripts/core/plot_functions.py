#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
(c) 2023 under a MIT License (https://mit-license.org)

Authors:
- Lukas Brunner || l.brunner@univie.ac.at

Abstract:

"""
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from collections import OrderedDict

import core.core_functions as cf
import core.dataset_functions as df


def plot_hyper_param(res, param, xscale='linear'):
    mean_train = res.cv_results_['mean_train_score']
    mean_test = res.cv_results_['mean_test_score']
    std_train = res.cv_results_['std_train_score']
    std_test = res.cv_results_['std_test_score']
    alphas = res.cv_results_[f'param_{param}'].filled().astype(float)

    fig, ax = plt.subplots()
    ax.set_xscale(xscale)

    ax.plot(alphas, mean_train, label='train')
    ax.plot(alphas, mean_test, label='validation')
    ax.fill_between(
        alphas,
        mean_train - std_train,
        mean_train + std_train,
        alpha=0.2)
    ax.fill_between(
        alphas,
        mean_test - std_test,
        mean_test + std_test,
        alpha=0.2)
    ax.scatter(
        float(res.best_params_[param]),
        mean_test.max(),
        marker='x', c='r')

    ax.grid()
    ax.set_xlabel(param)
    ax.set_ylabel('Score')
    ax.set_title('{}={:.4f}'.format(param, res.best_params_[param]))
    ax.legend()


def plot_coef_map(model):
    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(subplot_kw={'projection': proj}, figsize=(9, 4))

    coefs = model.coef_
    ds = cf.load_example_sample()

    try:
        coefs_2d = coefs.reshape(ds['lat'].size, ds['lon'].size)
    except ValueError:
        nan_mask = cf.get_land_mask()
        coefs_full = np.empty_like(nan_mask) * np.nan
        coefs_full[~nan_mask] = coefs[0]
        coefs_2d = coefs_full.reshape(ds['lat'].size, ds['lon'].size)

    max_ = np.nanmax([np.abs(coefs.min()), coefs.max()])

    map_ = ax.pcolormesh(
        ds['lon'], ds['lat'],
        coefs_2d * -1,
        cmap='RdBu_r',
        vmin=-max_, vmax=max_,
    )
    ax.coastlines()

    ax.set_xticks([-180, -120, -60, 0, 60, 120, 180])
    ax.set_yticks([-90, -60, -30, 0, 30, 60, 90])
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.set_xlabel('')
    ax.set_ylabel('')

    ax.set_title('Regression coefficients')

    fig.colorbar(map_, ax=ax, pad=.01)
    fig.tight_layout()
    return ax


def plot_map(da, **kwargs):
    kwargs_default = {
        'cbar_kwargs': {'pad': .01},
    }
    kwargs_default.update(kwargs)

    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(subplot_kw={'projection': proj}, figsize=(9, 4))
    da.plot.pcolormesh(ax=ax, **kwargs_default)

    ax.coastlines()
    ax.set_xticks([-180, -120, -60, 0, 60, 120, 180])
    ax.set_yticks([-90, -60, -30, 0, 30, 60, 90])
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.set_xlabel('')
    ax.set_ylabel('')

    fig.tight_layout()

    return ax


def plot_reliability_diagram(model, XX, yy):
    yp = model.predict(XX)  # predicted category
    yy_correct = (yp == yy).astype(int)  # correct predictions
    pp = model.predict_proba(XX)  # assigned probabilities
    pp_predicted = np.max(pp, axis=1)  # probability for assigned category
    # pp_correct = pp.swapaxes(0, 1)[yy]  # probability for correct category

    bin_size = .1
    # NOTE: np.digitize returns 0 for values < np.min(bins)
    # and 1 for values within the first bin
    bins = np.arange(bin_size, 1 + bin_size, bin_size)
    bins_center = np.arange(bin_size / 2, 1, bin_size)
    bins_idx = np.digitize(pp_predicted, bins)

    binned_counts = [(idx == bins_idx).sum() for idx in range(len(bins))]
    binned_accuracy = [yy_correct[idx == bins_idx].sum() / binned_counts[idx] if binned_counts[idx] > 0 else np.nan
                       for idx in range(len(bins))]
    binned_confidence = [pp_predicted[idx == bins_idx].mean() if binned_counts[idx] > 0 else np.nan
                         for idx in range(len(bins))]

    fig, ax = plt.subplots()
    ax.grid(axis='y', zorder=0)
    ax.bar(bins_center, binned_accuracy, width=.1, edgecolor='k', zorder=9)
    ax.plot([0, 1], [0, 1], color='k', label='Perfect reliability', zorder=99)
    ax.set_yticks(bins)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.set_aspect('equal')

    ax.set_xlabel('Confidence')
    ax.set_ylabel('Accuracy')

    for idx, count in enumerate(binned_counts):
        if count == 0:
            continue
        ax.annotate(
            # limit flaot to 4 digits
            str(count/sum(binned_counts)*100)[:4].rstrip('.') + '%',
            (bins_center[idx], .01),
            ha='center', fontsize='small', zorder=999)
        ax.vlines(binned_confidence[idx], .05, np.max([binned_accuracy[idx], .05]),
                  color='k', ls=':', zorder=999)

    ax.vlines([], [], [],  color='k', ls=':', label='Bin average confidence')
    ax.legend(loc='upper left')
    plt.tight_layout()

    return ax


def plot_probabilities_model(
    y_pred: list,
    dataset_names: list=None,
    whis: tuple=(0, 100),
    add_empty: bool=False,
    show_legend: bool=False,
    ypos_legend: float=6.,
    ax=None
) -> plt.axes:
    """
    Boxplot of probabilities per model variant.

    Parameters
    ----------
    y_pred : np.ndarray, shape (2, N)
        Predictied probability
    dataset_names : np.ndarray, shape (N,)
        List of dataset names
    whis : tuple, shape (2,), optional
        Extent of the whiskers
    show_legend : bool, optional
        How a legend of the box extent
    """
    if isinstance(y_pred, dict):
        binned = y_pred
    else:
        binned = cf.bin_by(y_pred[:, 1], dataset_names)

    # NOTE: the boxplot is sorted from bottom to top so
    # order alphabetically and reverse...
    binned = OrderedDict(sorted(binned.items())[::-1])
    # ...then move observations to the end
    for obs in df.observation_names[::-1]:
        if obs in binned.keys():
            binned.move_to_end(obs)

    model_labels = {idx: dataset for idx, dataset in enumerate(binned.keys()) if dataset in df.model_names}
    obs_labels = {idx: dataset for idx, dataset in enumerate(binned.keys()) if dataset in df.observation_names}

    fig, ax = plt.subplots(figsize=(5, 8))

    if show_legend:
        ypos = ypos_legend
        min_ = .57
        max_ = .92
        ax.boxplot(
            np.linspace(min_, max_, 1000),
            positions=[ypos],
            widths=.6,
            whis=whis,
            labels=None,
            vert=False,
            flierprops=dict(
                marker='o',
                markersize=1,
                markerfacecolor='k',
                markeredgecolor='k'
            )
        )
        ax.text(min_, ypos + .3, 'min', ha='center', va='bottom')
        ax.text(.25 * (3 * min_ + max_), ypos - .6, '25%', ha='center', va='top')
        ax.text(.5 * (min_ + max_), ypos + .3, '50%', ha='center', va='bottom')
        ax.text(.25 * (min_ + 3 * max_), ypos - .6, '75%', ha='center', va='top')
        ax.text(max_, ypos + .3, 'max', ha='center', va='bottom')
        ax.vlines(.5, ymin=-.5, ymax=len(binned) - .5, color='k', ls=':', lw=.5)
    else:
        ax.vlines(.5, ymin=-.5, ymax=len(binned) - .5, color='k', ls=':', lw=.5)

    ax.boxplot(
        binned.values(),
        positions=range(len(binned)),
        vert=False,
        whis=whis,
        flierprops=dict(
            marker='o',
            markersize=1,
            markerfacecolor='k',
            markeredgecolor='k'
        ))

    # --- x-axis ---
    ax.set_xlabel('Probability to be an observation')
    ax.set_xticks(np.arange(0,  1.1, .25))
    ax.set_xlim(-.01, 1.01)

    # --- y-axis ---
    ax.set_yticks([*model_labels.keys()])
    ax.set_yticklabels([*model_labels.values()])
    ax.tick_params(axis="y", length=0)
    try:
        ax.vlines(0, np.min([*model_labels.keys()]) - .5,
                  np.max([*model_labels.keys()]) + .5, lw=.75, color='k')
        ax.vlines(0, np.min([*obs_labels.keys()]) - .5,
                  np.max([*obs_labels.keys()]) + .5, lw=.75, color='k', ls=':')
    except ValueError:
        ax.vlines(0, 0, len(binned), lw=.75, color='k')
    ax.set_ylabel('Models', fontsize='x-large')

    ax2 = ax.secondary_yaxis('right')
    ax2.set_yticks([*obs_labels.keys()])
    ax2.set_yticklabels([*obs_labels.values()])
    ax2.tick_params(axis="y", length=0)
    ax2.set_ylim(np.min([*obs_labels.keys()]) - .5, np.max([*obs_labels.keys()]) + .5)
    try:
        ax.vlines(1, np.min([*obs_labels.keys()]) - .5, np.max([*obs_labels.keys()]) + .5, lw=.75, color='k')
        ax.vlines(1, np.min([*model_labels.keys()]) - .5, np.max([*model_labels.keys()]) + .5, lw=.75, color='k', ls=':')
    except ValueError:
        ax.vlines(1, 0, len(binned), lw=.75, color='k')
    ax2.set_ylabel(''.join([' '] * 92) + 'Observations', fontsize='x-large')

    ax.spines.top.set_visible(False)
    ax.spines.left.set_visible(False)
    ax.spines.right.set_visible(False)
    ax2.spines.right.set_visible(False)

    ax.set_ylim(-1.5, None)

    plt.tight_layout()

    return ax
