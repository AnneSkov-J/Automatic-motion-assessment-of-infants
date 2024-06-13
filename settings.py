#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: Settings
Description: This file loads packages and define root paths

Created on Thu Dec 21 17:38:17 2023
@author: anne
"""

#%% Load packages

import os
import csv
 
#For data strucure
import numpy as np
import pandas as pd
import scipy.io
import pickle

#For plots
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

#Others
import random
import math
import statistics as stat 
import scipy.stats as stats
from scipy.spatial import ConvexHull #Calculate area of movement (feature)
#pip install scikit-fuzzy
import skfuzzy as fuzz               #Fuzzy clustering method for clustering movements in one recording
from sklearn.cluster import KMeans   #K-means clustering method for clustering movements in one recording
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs
import time
import shutil                       #Moving files from directory to directory

#Regression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.svm import SVR, SVC
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import LeaveOneOut

#Segmentation
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from collections import Counter
from scipy.fft import fft #Fourier transformation

#Interpolation
from scipy.interpolate import interp1d

#Clustering
from sklearn.cluster import KMeans
from scipy.spatial import distance
from scipy.stats import chi2
from sklearn.mixture import GaussianMixture



#%% Define paths

## Root path
path_root = '/Users/anne/Library/CloudStorage/OneDrive-Personal/Documents, ONE/DTU/12. Semester (master thesis)/2.Python'

#Set working directory
os.chdir(path_root)

## Data path
path_data = os.path.join(path_root, '1.Data')
path_data_py = os.path.join(path_data, '2.Data_py/')
path_data_ML = os.path.join(path_data, '3.Data_ML_py/')
path_agedata = os.path.join(path_data, '4.Grouped data')
