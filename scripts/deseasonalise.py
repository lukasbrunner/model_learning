#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
(c) 2023 under a MIT License (https://mit-license.org)

Authors:
- Lukas Brunner || l.brunner@univie.ac.at

Abstract:
"""
import os
import sys
import numpy as np
import xarray as xr
from glob import glob
sys.path.append('../')

import core.core_functions as cf


def rolling_mean_season(ds, window=31):
    """Calculate mean seasonal cycle."""
    ds = ds.groupby('time.dayofyear').mean('time', keep_attrs=True)
    if window == 1:
        return ds
    days = ds['dayofyear'].size
    # extend beginning and end of the year by half a window size
    ds_end = ds.isel(dayofyear=np.arange(days - (window - 1)//2, days))
    ds_end = ds_end.assign_coords({'dayofyear': ds_end['dayofyear'] - days})
    ds_start = ds.isel(dayofyear=np.arange((window - 1)//2))
    ds_start = ds_start.assign_coords({'dayofyear': ds_start['dayofyear'] + days})
    ds = xr.concat([ds_end, ds, ds_start], dim='dayofyear') 
    ds = ds.rolling(dayofyear=window, center=True).mean()
    ds = ds.sel(dayofyear=np.arange(1, days+1))
    return ds


def main():
    for fn in glob(os.path.join(cf.BASEPATH['absolute_historical'], '*.nc')):
        if os.path.isfile(os.path.join(cf.BASEPATH['deseas_historical'], os.path.basename(fn))):
            continue
        
        ds = xr.open_dataset(fn, use_cftime=True)
        
        dataset_name = os.path.basename(fn).split('_')[2]
        varn = cf.varn_map.get(dataset_name, 'tas')
        ds = ds[varn]

        # NOTE: use only the training period for the calculation of the seasonal cycle
        ds_seas = rolling_mean_season(ds.sel(time=slice('1982', '2001')))
        
        ds_list = []
        for idx_group, ds_group in ds.groupby('time.dayofyear'):
            ds_group = ds_group - ds_seas.sel(dayofyear=idx_group)
            ds_list.append(ds_group)
        ds_deseas = xr.concat(ds_list, dim='time')
        ds_deseas = ds_deseas.sortby('time')  # need to sort again otherwise slicing will fail

        # NOTE: remove 366th day of leap years
        ds_deseas = ds_deseas.sel(time=~(ds_deseas['time.dayofyear'] == 366))
        
        # restore attributes
        ds_deseas.attrs = ds.attrs
        ds_deseas = ds_deseas.to_dataset()
        ds_deseas.to_netcdf(os.path.join(cf.BASEPATH['deseas_historical'], os.path.basename(fn)))

if __name__ == '__main__':
    main()