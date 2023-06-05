# Model Learning: disdinguish models and observations based on daily output maps

**Model learning** combines the terms _climate model_ and _machine learning_ providing a framework to disdinguish models and observations based on output maps. It draws on the concepts of model performance ("how different is a model from the observations?") and model independence ("how different are models from each other?"). The repository contains different machine learning classifiers to disdinguish models from observations (binary classification) and models from each other (multi-class classification). 

Compared to traditional approaches the use of data-driven machine learning approaches allows to use considerably shorter time periods as basis for the classification and results hold for out-of-sample datasets. One boiled down question we ask is: 

**Is a gridded map of daily temperature more likely to come from a model or from an observation?** 

![figure](plots/examples/examples_nolabel.gif)

Table of contents 
-----------------   

- [Model Learning: disdinguish models and observations based on daily output maps](#model-learning-disdinguish-models-and-observations-based-on-daily-output-maps)
  - [Table of contents](#table-of-contents)
  - [Overview](#overview)
    - [Terminology](#terminology)
    - [Structure of the repository](#structure-of-the-repository)
    - [Input data](#input-data)
  - [Example cases for `binary_logistic_regression`](#example-cases-for-binary_logistic_regression)
    - [Using pre-trained classifiers](#using-pre-trained-classifiers)
    - [Some results (presented at EGU 2023)](#some-results-presented-at-egu-2023)
    - [Training a new classifier](#training-a-new-classifier)
    - [Regularization](#regularization)
  - [About the included land sea mask](#about-the-included-land-sea-mask)

Overview
--------

The scientific and methodological details are discussed in the paper: Brunner and Sippel (accepted): Identifying climate models based on their daily output using machine learning. _Env. Data Sci._ Preprint: https://doi.org/10.31223/X53M0J

You are free to reuse this code for your own research following the conditions outlinend in the [license](./LICENSE). If you do so, please cite the paper above. I try to keep track of people working with my code so I would be happy if you'd also let me know when using my code or in case you have any questions of course: [email](mailto:l.brunner@univie.ac.at). 

### Terminology
To avoid confusion between different terms this document uses the following conventions:
- **model** exclusively refers to physical climate models which provide daily output maps used as input for the machine learning classifiers
- **classifier** refers to the machine learning algorithms used to distinguish models from observations

### Structure of the repository

The repository contains the following folders:
- `data`: contains the training and test data as well as several pre-trained classifiers (most data have to be downloaded separately)
- `plots`: is the default folder to save figues for the scripts and contains some examples
- `scripts`: contains the scripts used to train and test the classifiers
  - `core`: contains the core functions used by all scripts
  - `binary_logistic_regression`: **This is the recommended starting point** as it represents to simplest and most extensively documented case. It contains the scripts used to train and test the binary case (models versus observations) based on logistic regression classifiers.
  - `binary_cnn`: contains scripts based on convolutional neural network classifiers.
  - `multi_class_cnn`: contains scripts based on convolutional neural network classifiers for the multi-class case (recognize each dataset by its name).

### Input data

The default input data format are daily maps of 2-m surface air temperature on a 2.5x2.5 degree latitude-longitude grid (regridded with `cdo rempcon2`) but other frequencies, variables, or resolutions are inprinciple also possible. We use two different pre-processing steps:
- absolute fields
- deseasonalised fields: mean seasonal cycle removed from each day of the year, grid cell and dataset separately (see Brunner and Sippel (in review) for details)

Two on-the-fly pre-processing steps are also available:
- land masked (lm) fields
- fields with daily global mean removed (gm)

#### Data availability

The data used in for training and testing the classifier are quite large and I do not want to duplicate all of them, since most of them are available online. As a compromise please find the CMIP6 model data on [ESGF](https://esgf-node.llnl.gov/search/cmip6/) and all other datasets on [Zenodo](10.5281/zenodo.7998437). 

Example cases for `binary_logistic_regression`
----------------------------------------------

The folder contains three example workflows. For all cases ocean temperature grid cells (land masked - lm) are used as features with the mean over all features subtracted (daily global mean removed - gm). Trained classifiers are saved in the `./data/trained_classifiers` folder. 

1.  Training and testing on all datasets but in different time periods (temporally out-of-sample). 
    - `./binary_logreg_fit.ipynb`
    - `./binary_logreg_predict.ipynb`
2.  Training only on datasets not used for testing (dataset out-of-sample) iteratively. Note that this leads to a different classifier for each group of test datasets. See Brunner and Sippel (in review) for details on the grouping of the datasets. 
    - `./binary_logreg_dataset_outsample.py`
    - `./binary_logreg_dataset_outsample_plot.ipynb`
3. Testing the trained classifiers on an arbitrary dataset. This is mainly intended for initial testing and playing around with new datasets.   
    - `./binary_logreg_predict_single.ipynb`

### Using pre-trained classifiers

The repository contains pre-trained classifiers which can be used for testing arbitrary new datasets and/or variables. For the binary case (models versus observations) and logistic regression these are:

- `./trained_classifiers/binary_logreg_absolute_historical_lm_gm.sav`: a logistic regression classifier trained on all available datasets (see 1. above). 
- `./trained_classifiers/outsample/binary_logreg_absolute_historical_lm_gm_*.sav`: logistic regression classifiers trained on all datasetes except the one indicated in the filename (see 2. above).

You can load any dataset and use the `preprocess` function to prepare it for classification. The dataset should have the following properties:
- 3D array with dimensions (time, lat, lon)
- the variable should be daily temperature (although there also is skill for some related variables such as maximum temperature)
- the spatial resolution should be 2.5x2.5 degrees (to regrid from another resolution use, e.g., `cdo remapbil,data/land_mask.nc input.nc output.nc`)

For more information follow the examples given in `./binary_logreg_predict_single.ipynb`. 

![figure](plots/examples/examples_binary.gif)

### Some results ([presented at EGU 2023](https://meetingorganizer.copernicus.org/EGU23/EGU23-492.html))

![figure](poster_EGU23.jpg) 

### Training a new classifier

For an example on how to train a new classifier follow the steps in `./binary_logreg_fit.ipynb`. You could try, e.g., to set the `land_mask` to `False` to see how the classification performance changes when using all grid cells. 

### Regularization

If you want to speed up the training process you can set the `C` parameter in the `LogisticRegression` to a fixed value instead of optimising it using cross-validation. For the case of global daily temperature an sensible order of magnitude is `1e-2` (the lm-gm case uses `0.002`). 

## About the included land sea mask

The land sea mask aims to fulfill two requirements:
- mask out ocean grid cells
- mask out grid cells not available in one of the datasets (the only dataset used which does not have all grid cells available is IOSST). 

Therefore, it combines two masks, one based on the Python `regionmask` package (drawing on natural earth polygons) and one based on missing values in the IOSST dataset (see `./create_land_mask.ipynb`).