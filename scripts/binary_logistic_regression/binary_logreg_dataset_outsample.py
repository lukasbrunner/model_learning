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
import pickle
from sklearn.linear_model import LogisticRegression

import core.core_functions as cf
import core.dataset_functions as df

random_init = 651
land_masked = True
global_mean = True


def read_trainset():
    samples_model = cf.get_samples(
        period=slice('1982', '2001'),
        land_masked=land_masked,
        global_mean=global_mean,
        time_steps=200,
        random_init=random_init,
        datasets=df.model_names,  # 43 models
    )

    samples_obs = cf.get_samples(
        period=slice('1982', '2001'),
        land_masked=land_masked,
        global_mean=global_mean,
        time_steps=2150,  # 200*43/4
        random_init=random_init,
        datasets=df.observation_names,  # 4 observations
    )

    return xr.concat([samples_model, samples_obs], dim='sample')


def fit_predict(trainset, testset, test_datasets=None):

    # index to select trainsamples not in test_datasets and...
    idx_train = np.isin(trainset['dataset_name'], test_datasets, invert=True)
    # ...testsamples in test_datasets
    idx_test = np.isin(testset['dataset_name'], test_datasets)

    X_train = trainset.values[idx_train]
    y_train = df.get_category_ids(trainset['dataset_name'].values[idx_train])

    X_test = testset.values[idx_test]
    y_test_name = testset['dataset_name'].values[idx_test]
    # y_test = df.get_category_ids(y_test_name)

    nan_mask = np.any(np.isnan(X_train), axis=0)
    X_train = X_train[:, ~nan_mask]
    X_test = X_test[:, ~nan_mask]

    logreg = LogisticRegression(
        penalty='l2',
        C=.002,  # set as in Brunner and Sippel (2023)
        solver='liblinear',
    )

    logreg.fit(X_train, y_train)

    # save each trained classifier
    savename = 'logreg{}_{}.sav'.format(
        ('_lm' if land_masked else '') + ('_gm' if global_mean else ''),
        '-'.join(test_datasets))
    pickle.dump(logreg, open(os.path.join(
        '../../data/trained_classifiers/outsample',
        savename), 'wb'))

    probability = logreg.predict_proba(X_test)
    # write the results for each dataset in a list
    # note that multiple datasets can be used in the testset at
    # the same time if they are in the same family. This saves
    # time (i.e., we do not need to train a classifier for each
    # dataset but just for each group).
    return cf.bin_by(probability[:, 1], y_test_name)


def main():
    trainset = read_trainset()
    testset = cf.get_samples(
        period=slice('2005', '2014'),
        land_masked=land_masked,
        global_mean=global_mean,
    )

    datasets_in_family = {}
    for key, value in df.dataset_families.items():
        datasets_in_family[value] = datasets_in_family.get(value, []) + [key]

    # NOTE: we assume here that all datasets are used!
    # otherwise we need to check if the test_datasets are in trainset and testset
    probabilities = {}
    for family, datasets in datasets_in_family.items():
        if len(datasets) == 0:
            continue

        print('Dataset group: {}'.format(family))
        print(f'Datasets excluded from training: {", ".join(datasets)}')
        proba = fit_predict(trainset, testset, test_datasets=datasets)

        probabilities.update(proba)

    savename =  'binary_logreg_dataset_outsample{}{}.sav'.format(
        dataset_type,
        ('_lm' if land_masked else '') + ('_gm' if global_mean else ''),
    )
    pickle.dump(probabilities, open(os.path.join(
        '../../data/trained_classifiers',
        savename), 'wb'))


if __name__ == '__main__':
    main()
