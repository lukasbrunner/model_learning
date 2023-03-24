#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
(c) 2023 under a MIT License (https://mit-license.org)

Authors:
- Lukas Brunner || l.brunner@univie.ac.at

Abstract:

"""
import os
import numpy as np
import xarray as xr
from glob import glob
from collections import OrderedDict

BASEPATH = '/jetfs/home/lbrunner/ml_logreg_data'
varn_map = {
    'ERA5': 't2m',
    'MERRA2': 'T2M',
    'IOSST': 'sst',
    '20CR': 'air',
}


def select_time_steps(
        ds: xr.DataArray,
        nr: int,
        how: str,
        random_init: int=None,
        idx: int=None) -> xr.DataArray:
    """Select a number of time steps using different methods.

    Paramerters
    -----------
    ds : xr.DataArray
    nr : int
        Number of time steps to select. If None all time steps will be selected and
        `how` will be disregarded.
    how : str, one of {'random', 'first', 'last'}
        Strategy to select timesteps.
    random_init : int, optional, by default None
        If not None used to initialise the random state.

    Returns
    -------
    xr.DataArray
    """
    if nr is None:
        return ds

    if ds['time'].size < nr:
        raise ValueError(f'time_steps exceeds times steps {ds["time"].size} < {nr}')

    if how == 'random':
        if random_init is not None:
            # different seed for each model (but constant in consequtive calls)
            random_init += idx
        idx_sel = np.sort(np.random.RandomState(random_init).randint(ds['time'].size, size=nr))
    elif how == 'first':
        idx_sel = range(nr)
    elif how == 'last':
        idx_sel = range(ds['time'].size - nr, ds['time'].size)
    else:
        raise ValueError(f'{how=} not allowed')

    return ds.isel(time=idx_sel)


def get_filenames(models: list) -> list:
    if models is None:
        return sorted(glob(os.path.join(BASEPATH, '*.nc')))
    return sorted([glob(os.path.join(BASEPATH, f'tas_day_{model}_*.nc'))[0]
                   for model in np.atleast_1d(models)])


def area_weighted_mean(ds: xr.DataArray):
    """Calculate an area weighted mean (approx. by cosine of the latitude)."""
    weights_lat = np.cos(np.radians(ds['lat']))
    return ds.weighted(weights_lat).mean(['lat', 'lon'], keep_attrs=True)


def get_land_mask():
    land_mask = xr.open_dataset(os.path.join('data', 'land_mask.nc'))['land_mask']  # true on land
    return land_mask


def preprocess(ds: xr.DataArray, land_masked: bool, global_mean: bool):
    if land_masked:
        mask = get_land_mask()
        ds = ds.where(~mask)  # inverse to create index
    if global_mean:
        ds = ds - area_weighted_mean(ds)
    return ds


def load_example_sample(dataset=None) -> xr.DataArray:
    fn = get_filenames(dataset)[0]
    ds = xr.open_dataset(fn).isel(time=0)
    ds = ds.drop('height', errors='ignore')
    if 'zlev' in ds:
        ds = ds.squeeze('zlev', drop=True)
    dataset_name = os.path.basename(fn).split('_')[2]
    varn = varn_map.get(dataset_name, 'tas')
    da = ds[varn]
    da.name = 'tas'
    return da


def load_dataset(fn: str) -> xr.DataArray:
    ds = xr.open_dataset(fn, use_cftime=True)
    ds = ds.drop('height', errors='ignore')
    if 'zlev' in ds:
        ds = ds.squeeze('zlev', drop=True)
    dataset_name = os.path.basename(fn).split('_')[2]
    ds = ds.expand_dims({
        'dataset_name': [dataset_name],
    })
    # the observations are not cmorized -> map variable names
    varn = varn_map.get(dataset_name, 'tas')
    da = ds[varn]
    da.name = 'tas'
    return da


def get_samples(
        period,
        land_masked: bool=True,
        global_mean: bool=True,
        time_steps: int=None,
        time_steps_select: str='random',
        random_init: int=None,
        datasets: list=None,
        verbose: bool=False,
) -> xr.DataArray:
    """Load and preprocess training and test samples from different datasets.

    Parameters
    ----------
    period : slice
        Time period to use as slice of two strings (e.g., slice('1991', '2000')).
    land_masked : bool, optional, by default True
        Whether to mask out land grid cells in the samples.
    global_mean : bool, optional, by default True
        Whether to remove the global mean value from each sample.
    time_steps : integer, optional, by default None
        Time steps to select per dataset. If not None it has to be an integer
        smaller or equal to the number of time steps in the shortest dataset.
    select_time_steps : string, optional
        Only valid if time_steps is not None. Specifies how time steps
        are selected
        - 'random' (default): random time steps will be selected
        - 'first': the first time_steps time steps will be selected
        - 'last': the last time_steps time steps will be selected
    random_init : integer
        Only valid if select_time_steps is 'random'
    datasets : list, optional
        If not None use only the given datasets instead of all.

    Returns
    -------
    samples : xr.DataArray
        A 2-dimensional data array with dimensions (sample, features)
    """
    filenames = get_filenames(datasets)

    if not land_masked and np.any(['IOSST' in fn for fn in filenames]):
        raise ValueError('IOSST should only be used for land masked case!')

    if verbose:
        print('Number of datasets:', len(filenames))

    ds_list = []
    for idx, fn in enumerate(filenames):
        ds = load_dataset(fn)
        if verbose:
            print('Loaded dataset', ds['dataset_name'].values[0])
        ds = ds.sel(time=period)
        ds = select_time_steps(ds, time_steps, time_steps_select, random_init, idx)
        ds = preprocess(ds, land_masked, global_mean)

        ds = ds.stack(sample=['dataset_name', 'time'])
        ds = ds.stack(feature=['lat', 'lon'])

        ds_list.append(ds)

    ds = xr.concat(ds_list, dim='sample')
    if verbose:
        print('Number of samples:', ds['sample'].size)
    return ds


def bin_by(y_pred: list, groups: list):
    """Bin values into groups.

    Each value in y_pred will be attributed to the coresponding element in
    groups. Identical elements in groups will be grouped together. Can be used
    to group predicted probabilities by model or day of the year.

    Parameters
    ----------
    y_pred : list, shape (M,)
    groups : list, shape (M,)

    Returns
    -------
    nested list, shape (N<=M, x)"""
    groups_unique = np.unique(groups)[::-1]
    bins = OrderedDict([(key, []) for key in groups_unique])
    for yy, gg in zip(y_pred, groups):
        bins[gg].append(yy)

    # move the observational datasets to the end (will appear first in plot)
    for obs in ['IOSST', '20CR', 'MERRA2', 'ERA5']:
        try:
            bins.move_to_end(obs)
        except KeyError:
            pass

    if 366 in bins.keys():
        bins.pop(366)  # remove last day of year for leap years

    return bins
