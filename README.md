# Automatic-motion-assessment-of-infants
Modeling movement patterns at different developmental stages during the first 6 month of life using machine learning and 3D motion caption data

## Data preparation

**File: 0.load_raw_data.py**

The file loads raw mat-data files and store data in directories for later use in python
Furthermore, the file fiters the data according to age

## Age Prediction Model

**File: 1.feature_extraction.py**

The files calcualtes features over data and save them in directories
The dictionaries are seperated into a training data set and a hold-out data set and saved accordingly

**File: 1.regression_models.py**

First, the file prepared data for regression

The file extract selected features 
The features from each file are combined into a data matrix. The data in the matrix is interpolated and outliers are removed.

Secondly, the file use regression models to predict the age from the seelcted features

The file first fits different regression models to the data and secondly uses a Random Forest regressor as model

## Motion Model

**File: 2.group_data.py**

The files group data in age-groups and combine data-files to one for each age-group

**File: 2.segmentation.py**

The file contain the four segmentation methods: frequent pose, zero velocity, PCA and PPCA

**File: 2.clustering.py**

The file contain three clustering methods: K-Means, K-Means with Mahalanbis, and GMM


## Note
All scripts are raw scripts with comments and describtions from the author. 
