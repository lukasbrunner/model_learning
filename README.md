# Model Learning: disdinguish daily model output and observations using logisitic regression

**Model learning** combines the terms _climate model_ and _machine learning_ providing a fremework to disdinguish models and observations based on their daily output maps using different machine learning classifiers. This repository useses **logistic regression** to classify maps of daily termperature as coming from observational datasets or climate models.  

## Overview

The scientific and methodological details are discussed in the paper: Brunner and Sippel (in review): Identifying climate models based on their daily output using machine learning. _Env. Data Sci._ Preprint: https://doi.org/10.31223/X53M0J

You are free to reuse this code for your own research following the conditions outlinend in the [license](./LICENSE). If you do so, please cite the paper above. For any questions feel free to contact me via [email](mailto:l.brunner@univie.ac.at). 

## Important terminology

To avoid confusion between different terms this document uses the following conventions:
- **model** refers to physical climate models which provide daily output maps used as input for the machine learning classifiers
- **classifier** refers to the machine learning algorithm used to distinguish models from observations

## Example cases

The repository contains three example workflows. For all cases ocean temperature grid cells (land masked - lm) are used as features with the mean over all features subtracted (global mean removed - gm). Trained classifiers are saved in the `./trained_classifiers` folder. 

1.  Training and testing on all datasets but in different time periods (temporally out-of-sample). 
    - `./logreg_fit.ipynb`
    - `./logreg_predict.ipynb`
2.  Training only on datasets not used for testing (dataset out-of-sample) iteratively. Note that this leads to a different classifier for each group of test datasets. See Brunner and Sippel (in review) for details on the grouping of the datasets. 
    - `./logreg_dataset_outsample.py`
    - `./logreg_dataset_outsample_plot.ipynb`
3. Testing the trained classifiers on an arbitrary dataset. This is mainly intended for initial testing and playing around with new datasets.   
    - `./logreg_predict_single.ipynb`

## Using pre-trained classifiers

The repository contains pre-trained classifiers for two cases which can be used for testing arbitrary new datasets and/or variables. 

- `./trained_classifiers/logreg_lm_gm.sav`: a logistic regression classifier trained on all available datasets (see 1. above). 
- `./trained_classifiers/outsample/logreg_lm_gm_*.sav`: logistic regression classifiers trained on all datasetes except the one indicated in the filename (see 2. above).

You can load any dataset and use the `preprocess` function to prepare it for classification. The dataset should have the following properties:
- 3D array with dimensions (time, lat, lon)
- the variable should be daily temperature (although related variables such as maximum temperautre also work)
- the spatial resolution should be 2.5x2.5 degrees (to regrid from another resolution use, e.g., `cdo remapbil,data/land_mask.nc input.nc output.nc`)

For more information follow the examples given in `./logreg_predict_single.ipynb`. 

## Training a new classifier

For an example on how to train a new classifier follow the steps in `./logreg_fit.ipynb`. You could try, e.g., to set the `land_mask` to `False` to see how the classification performance changes when using all grid cells. 

### Regularization

If you want to speed up the training process you can set the `C` parameter in the `LogisticRegression` to a fixed value instead of optimising it using cross-validation. For the global case an sensible order of magnitude is `1e-2` (the lm-gm case uses `0.002`). 

## About the land sea mask

The land sea mask aims to fulfill two requirements:
- mask out ocean grid cells
- mask out grid cells not available in one of the datasets (the only dataset used which does not have all grid cells available is IOSST). 

Therefore, it combines two masks, one based on the Python `regionmask` package (drawing on natural earth polygons) and one based on missing values in the IOSST dataset (see `./create_land_mask.ipynb`).