#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
(c) 2023 under a MIT License (https://mit-license.org)

Authors:
- Lukas Brunner || l.brunner@univie.ac.at

Abstract:

"""
model_names = [
  'ACCESS-CM2',
  'ACCESS-ESM1-5',
  'AWI-CM-1-1-MR',
  'AWI-ESM-1-1-LR',
  'BCC-CSM2-MR',
  'BCC-ESM1',
  'CAMS-CSM1-0',
  'CESM2',
  'CESM2-WACCM',
  'CMCC-CM2-HR4',
  'CMCC-CM2-SR5',
  'CMCC-ESM2',
  'CNRM-CM6-1',
  'CNRM-CM6-1-HR',
  'CNRM-ESM2-1',
  'CanESM5',
  'EC-Earth3',
  'EC-Earth3-AerChem',
  'EC-Earth3-Veg',
  'EC-Earth3-Veg-LR',
  'FGOALS-f3-L',
  'FGOALS-g3',
  'GFDL-CM4',
  'GFDL-ESM4',
  'HadGEM3-GC31-LL',
  'HadGEM3-GC31-MM',
  'INM-CM4-8',
  'INM-CM5-0',
  'IPSL-CM5A2-INCA',
  'IPSL-CM6A-LR',
  'KACE-1-0-G',
  'KIOST-ESM',
  'MIROC-ES2L',
  'MIROC6',
  'MPI-ESM-1-2-HAM',
  'MPI-ESM1-2-HR',
  'MPI-ESM1-2-LR',
  'MRI-ESM2-0',
  'NESM3',
  'NorESM2-LM',
  'NorESM2-MM',
  'TaiESM1',
  'UKESM1-0-LL',
]

observation_names = [
    '20CR',
    'ERA5',
    'IOSST',
    'MERRA2',
]

dataset_families = {
    'ACCESS-CM2': 'ACCESS',
    'ACCESS-ESM1-5': 'ACCESS',
    'AWI-CM-1-1-MR': 'AWI',
    'AWI-ESM-1-1-LR': 'AWI',
    'BCC-CSM2-MR': 'BCC',
    'BCC-ESM1': 'BCC',
    'CAMS-CSM1-0': 'CAMS-CSM1-0',  # one model family
    'CESM2': 'CESM',
    'CESM2-WACCM': 'CESM',
    'CMCC-CM2-HR4': 'CMCC',
    'CMCC-CM2-SR5': 'CMCC',
    'CMCC-ESM2': 'CMCC',
    'CNRM-CM6-1': 'CNRM',
    'CNRM-CM6-1-HR': 'CNRM',
    'CNRM-ESM2-1': 'CNRM',
    'CanESM5': 'CanESM5',  # one model family
    'EC-Earth3': 'EC-Earth',
    'EC-Earth3-AerChem': 'EC-Earth',
    'EC-Earth3-Veg': 'EC-Earth',
    'EC-Earth3-Veg-LR': 'EC-Earth',
    'FGOALS-f3-L': 'FGOALS',
    'FGOALS-g3': 'FGOALS',
    'GFDL-CM4': 'GFDL',
    'GFDL-ESM4': 'GFDL',
    'HadGEM3-GC31-LL': 'HadGEM',
    'HadGEM3-GC31-MM': 'HadGEM',
    'INM-CM4-8': 'INM',
    'INM-CM5-0': 'INM',
    'IPSL-CM5A2-INCA': 'IPSL',
    'IPSL-CM6A-LR': 'IPSL',
    'KACE-1-0-G': 'KACE-1-0-G',
    'KIOST-ESM': 'KIOST-ESM',  # one model family
    'MIROC-ES2L': 'MIROC',
    'MIROC6': 'MIROC',
    'MPI-ESM-1-2-HAM': 'MPI',
    'MPI-ESM1-2-HR': 'MPI',
    'MPI-ESM1-2-LR': 'MPI',
    'MRI-ESM2-0': 'MRI-ESM2-0',  # one model family
    'NESM3': 'NESM3',
    'NorESM2-LM': 'NorESM2',
    'NorESM2-MM': 'NorESM2',
    'TaiESM1': 'TaiESM1',
    'UKESM1-0-LL': 'HadGEM',
    # --- nominally also add observations ---
    'ERA5': 'ERA5',
    'MERRA2': 'MERRA2',
    '20CR': '20CR',
    'IOSST': 'IOSST',
}


def get_category_ids(dataset_names: list) -> list:
    def _get_category_id(dataset_name):
        if dataset_name in observation_names:
            return 1
        if dataset_name in model_names:
            return 0
        raise ValueError(dataset_name)
    if isinstance(dataset_names, str):
        return _get_category_id(dataset_names)
    return [_get_category_id(dataset_name) for dataset_name in dataset_names]


def get_categories(dataset_names: list) -> list:
    def _get_category(dataset_name):
        if dataset_name in observation_names:
            return 'observation'
        if dataset_name in model_names:
            return 'model'
        raise ValueError(dataset_name)
    if isinstance(dataset_names, str):
        return _get_category(dataset_names)
    return [_get_category(dataset_name) for dataset_name in dataset_names]


def get_groups(dataset_names: list) -> list:
    if isinstance(dataset_names, str):
        return dataset_families[dataset_names]
    return [dataset_families[dataset_name] for dataset_name in dataset_names]
