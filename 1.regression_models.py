#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: Age Prediction Model, Regression models
Description: 
This file prepares feature-data for a regression model 

Created on Wed May 15 14:31:58 2024
@author: anne
"""
#%% Load packages from setting
import os
path_root = '...'
os.chdir(path_root)
from settings import *
from functions import *

#%% Settings 

#Define the version of features to use
version = '100524' #date

#Define desired window-length for B features
window_select = '_30_' #or '_60_'

#%%Load files:
holdin = False #False
holdout = True #True

if holdin == True:
    #Load hold-in files
    path_file = os.path.join(path_data_ML, 'file_ls.txt')
    with open(path_file, 'r') as f:
        file_ls = f.readlines()
    file_ls = [int(line.strip()) for line in file_ls]

elif holdout == True:
    # Load hold-out files
    path_file = os.path.join(path_data_ML, '0.Hold_out_py', 'file_ls.txt')
    with open(path_file, 'r') as f:
        file_ls = f.readlines()
    file_ls = [int(line.strip()) for line in file_ls]

#%%################
# PREPARE DATA   ##
###################
"""The part of the file loads a dictionary with features for each sample/infant
Gather features of selected joints into one data matrix
Interpolates the window-features with 99th percentile of the window-lengths in data
Remove outliers from features
Finally, the first part of the file flatten and save the data matrix"""
    
#%% Gather all features for all patients in one data matrix

num_samples = len(file_ls)

#All features:
X_ls = []
cluster_mat = np.empty((num_samples, 1))       # 125 patients, 1 features
dist_cnt_mat = np.empty((num_samples, 6))     # 125 patients, 6 features
volume_mat = np.empty((num_samples, 6))       # 125 patients, 6 features

y = np.empty(num_samples)                      # 125 patients
ID = np.empty(num_samples)                     # 125 patients

#Index:
select_joints = ['Body', 'Head', 'RHand', 'LHand', 'RFoot', 'LFoot']

pos_joints  = ['BodyCenter','HeadCenter', 'RHand',      'LHand',      'RFoot',      'LFoot']        #Also used for volume
ang_joints  = ['BodyAngle', 'HeadAngle',  'RLArmAngle', 'LLArmAngle', 'RLLegAngle', 'LLLegAngle']   #Use lower arm and leg angles as substitutes for hand and foot angle

for i in range(num_samples):
    
    file = file_ls[i]
    file = f'{file:03d}'
    
    #Load feature-data
    path_file = os.path.join(path_feature, 'Feature_' + version + '_py', file + '.pkl')
    feature_dt = load_dict_func (file, path_file) 
    
    df = pd.DataFrame()
    dist_cnt_df = pd.DataFrame()
    volume_df = pd.DataFrame()
    cluster_df = pd.DataFrame()
    
    for key in feature_dt.keys():
        
        if window_select in key:
        
            #Position features
            if key.startswith('pos_'):
                data_dt = feature_dt[key]
                #Statistical features
                for stat_key in data_dt.keys():
                    data_df = data_dt[stat_key][pos_joints] #Extract relevant joints
                    data_df = data_df.drop(data_df.index[-1]) #Remove last row of data (nan-row)
                    if stat_key == 'entro':
                        data_df = data_df.fillna(0)
                    data_df.columns = [key[:-2] + stat_key + '_' + i for i in data_df.columns]
                    df = pd.concat([df, data_df], axis = 1) #Save
                
            #Angle features
            elif key.startswith('ang_'):
                data_dt = feature_dt[key]
                #Statistical features
                for stat_key in data_dt.keys():
                    data_df = data_dt[stat_key][ang_joints] #Extract relevant joints
                    data_df = data_df.drop(data_df.index[-1]) #Remove last row of data (nan-row)
                    if stat_key == 'entro' or '_60_' in key:
                        data_df = data_df.fillna(0)
                    data_df.columns = [key[:-2] + stat_key + '_' + i for i in data_df.columns]
                    df = pd.concat([df, data_df], axis = 1) #Save
            
            #Distance features
            elif key.startswith('dist_') and 'cnt' not in key:
                data_dt = feature_dt[key]
                #Statistical features
                for stat_key in data_dt.keys():
                    data_df = data_dt[stat_key][select_joints] #Extract relevant joints 
                    data_df = data_df.drop(data_df.index[-1]) #Remove last row of data (nan-row)
                    if stat_key == 'entro' or '_60_' in key:
                        data_df = data_df.fillna(0)
                    data_df.columns = [key[:-2] + stat_key + '_' + i for i in data_df.columns]
                    df = pd.concat([df, data_df], axis = 1) #Save
                    
            elif key.startswith('dist_') and 'cnt' in key:
                data_df = feature_dt[key][select_joints] #Extract relevant joints
                data_df = data_df.iloc[[0]]
                data_df.columns = [key[:-2] + i for i in data_df.columns]
                dist_cnt_df = pd.concat([dist_cnt_df, data_df], axis = 1) #Save
            
            elif key.startswith('volume_'):
                data_df = feature_dt[key][pos_joints] #Extract relevant joints
                data_df.columns = [key[:-2] + i for i in data_df.columns]
                volume_df = pd.concat([volume_df, data_df], axis = 1) #Save
                
            elif key.startswith('cluster_'):
                data = feature_dt[key]['n_cluster']
                data_df = pd.DataFrame({key[:-2] + 'n_cluster' : data})
                cluster_df = pd.concat([cluster_df, data_df], axis = 1) #Save

    X_ls.append(df)
    
    #Save odd-sized data
    dist_cnt_mat[i] = dist_cnt_df
    volume_mat[i] = volume_df
    cluster_mat[i] = cluster_df

    ### Age
    y[i] = feature_dt['Age'] 
    
    ### ID
    ID[i] = feature_dt['ID']
    
    nan_test = df.isna().any()
    if np.sum(nan_test) != 0:
        print('***')
        break
    
X_colnames = df.columns #210
dist_colnames = dist_cnt_df.columns #6
vol_colnames = volume_df.columns #6
cluster_colnames = cluster_df.columns #1
#%% Interpolate window-data

#Find interpolation window:
window_ls = []
for mat in X_ls:
    n_window = mat.shape[0]
    window_ls.append(n_window)
#Statistics on number of windows
np.mean(window_ls) 
np.percentile(np.array(window_ls), 99) 

if holdin == True:
    n_interpol_window = int(np.percentile(np.array(window_ls), 99)) #the number of windows to interpolate to 
elif holdout == True:
    if version == '100524':
        n_interpol_window = 1083
    elif version == '210524':
        n_interpol_window = 1083
        
matrix_shape = (n_interpol_window, 210)         
X_mat = np.empty((num_samples, *matrix_shape)) 

for i in range(num_samples):
    data_df = X_ls[i]
    
    ##Interpolation
    n_window = data_df.shape[0]
    
    #Create interpolation function
    x = np.arange(n_window) #Generate an array of index from 0 to n_window-1
    interpol_func = interp1d(x, data_df, axis=0, kind = 'linear') #axis = 0 == interpolate along first axis of data_df (rows)
    
    #Interpolate segment
    x_interpol = np.linspace(0, n_window - 1, n_interpol_window) #Generate n_interpol_window evenly spread numbers in the interval [0; n_window - 1]
    interpol_data = interpol_func(x_interpol)
    interpol_df = pd.DataFrame(data = interpol_data, columns = data_df.columns)
    X_mat[i] = interpol_data

X_shape = X_mat.shape     

#%% Remove outliers

threshold = 3 #threshold = number of standard deviations from mean
outlier_cnt = 0

#Loop over joints
for i in range(X_shape[1]):
    #Loop over features
    for j in range(X_shape[2]):
        data = X_mat[:,i,j]
        
        if np.std(data) != 0:
            #Z-score method
            z_scores = (data - np.mean(data)) / np.std(data)
            outlier_index = np.abs(z_scores) > threshold
            outlier_cnt += np.sum(outlier_index)
        else: 
            #No outliers if std dev is zero
            outlier_index = np.zeros_like(data, dtype=bool)  
        
        #Imputation with mean of data without outliers
        data_without = data[~outlier_index]
        mean = np.mean(data_without)
        data_imputed = np.where(outlier_index, mean, data)
        
        #Insert imputed data into X
        X_mat[:,i,j] = data_imputed        
    
#Number of outliers imputed (threshold = 3)
outlier_cnt 

#%% Flatten data
#Flatten X and add cluster-data
X_flat1 = X_mat.reshape(X_mat.shape[0], -1)
X_flat = np.hstack((X_flat1, dist_cnt_mat, volume_mat, cluster_mat))
colnames = list(X_colnames.values) + list(dist_colnames.values) + list(vol_colnames.values) + list(cluster_colnames.values) #223 features

#Shape:
X_flat.shape  #[125 patients, n_interpol_window * 210 joint-features + 6 dist_cnt features + 6 volume-features + 1 cluster-feautes = 227.443 feature-values pr patient]

#nan test
np.sum(np.isnan(X_flat))

#Save
if holdin == True:
    path_file = os.path.join(path_model, 'Data/Xflat_30_'+version+'.npy')
    np.save(path_file, X_flat)
    
    path_file = os.path.join(path_model, 'Data/y_'+version+'.npy')
    np.save(path_file, y)

elif holdout == True:
    path_file = os.path.join(path_model, 'Data/Xflat_HO_'+version+'.npy')
    np.save(path_file, X_flat)
    
    path_file = os.path.join(path_model, 'Data/y_HO_'+version+'.npy')
    np.save(path_file, y)
    
    
#%%##################
# REGRESSION MODEL ##
#####################
"""The part of the file tries different regression models on the data matrix
Fits a Random Forest Regressor to the data matrix
Use the Random Forest Regressor on selected features of the data matrix
Use the Random Forest Regressor on the data matrix seperated into age-groups
Use Backward selection when running the Random Forest Regressor on the data matrix
Finally, the second part of the file uses the Random Forest Regressor on the holdout data 
"""

cluster_colnames =  ['cluster_30_n_cluster']
#%% Try different regression models 

#Linear regression
Reg_func (X_flat, y, n_splits = 5, model = 'Linear', model_parameter = None) #model parameter = None

## KNN
Reg_func (X_flat, y, n_splits = 5, model = 'KNN', model_parameter = 5) #model parameter = k
Reg_func (X_flat, y, n_splits = 5, model = 'KNN', model_parameter = 25) #model parameter = k
#Loop over k values
KNNreg_func (X_flat, y, 5)

## SVM
Reg_func (X_flat, y, n_splits = 5, model = 'SVM', model_parameter = 'linear') #model parameter = kernel

lin_coef = SVMreg_func (X_flat, y, n_splits = 5, kernel = 'linear') #
Reg_func (X_flat, y, n_splits = 5, model = 'SVM', model_parameter = 'poly') #model parameter = kernel
SVMreg_func (X_flat, y, n_splits = 5, kernel = 'poly')
Reg_func (X_flat, y, n_splits = 5, model = 'SVM', model_parameter = 'rbf') #model parameter = kernel
SVMreg_func (X_flat, y, n_splits = 5, kernel = 'rbf')

#%% Random forest - all features

# Parameter estimation:
best_param = {'bootstrap': True,        #[default = True]
              'max_depth': None,        #Max depth (length of the longest path from root to leaf) of each the decision trees in the forest [default = None]
              'max_features': 'log2',   #Maximum number of feature selected (sqrt = sqrt(total number of features)) [default = 1.0, "sqrt", "log2", 0.5]
              'max_leaf_nodes': None,   #Maximum number of leaf nodes in each tree in the forest [default = None]
              'max_samples': None,      #Maximum number of samples used in random sampling. decimal = % of total number of samples [default = None, 0.1]
              'min_impurity_decrease': 0.0, #Minimum decrease in impurity needed for a node to split [default = 0.0, 0.5]
              'min_samples_leaf': 5,    #Minimum number of samples required to be at a leaf node [default = 1, 5, 10]
              'min_samples_split': 5,   #Minimum number of samples required to split a node (=reduce impurity) [default = 2, 5, 10]
              'n_estimators': 300,      #Number of trees in the forest [default = 100, 125, 150, 175, 200, 225, 250]
              'loss_function': 'squared_error'    #absolute_error, squared_error
}

#Extracting the feature coefficients
y_true, y_pred, coef = ForestRegCV_func(X_flat, y, n_splits = 5, param = best_param)

## Coefficients
#coef.shape # (227443,)
#Extract special-features:
coef_cluster_df = pd.DataFrame(data = [coef[-1]], columns = cluster_colnames)
coef = coef[:-1] #delete cluster 
coef_volume_df = pd.DataFrame(data = coef[-6:]).T
coef_volume_df.columns = vol_colnames
coef = coef[:-6] #delete volume 
coef_dist_df = pd.DataFrame(data = coef[-6:]).T
coef_dist_df.columns = dist_colnames
coef = coef[:-6] #delete dist 
coef_special_df = pd.concat([coef_dist_df, coef_volume_df, coef_cluster_df], axis = 1)
#Extract window-features
coef_mat = coef.reshape(1083, 210)
coef_w_df = pd.DataFrame(coef_mat, columns = X_colnames) #index = windows
coef_mean_df = pd.DataFrame(coef_w_df.mean(axis = 0)).T #mean over windows
coef_mean_df.columns = X_colnames
coef_sum_df = pd.DataFrame(coef_w_df.sum(axis = 0)).T #sum over windows
coef_sum_df.columns = X_colnames

#Save coefficients
path_file = os.path.join(path_root, '3.Model/Model_output/'+version+'/Coefficients_30.xlsx')
os.makedirs(os.path.dirname(path_file), exist_ok=True)
with pd.ExcelWriter(path_file, engine='openpyxl') as writer:
    coef_w_df.to_excel(writer, sheet_name='Window-features', index=False)
    coef_mean_df.to_excel(writer, sheet_name='Mean-features', index=False)
    coef_sum_df.to_excel(writer, sheet_name='Sum-features', index=False)
    coef_dist_df.to_excel(writer, sheet_name='Dist-features', index=False)
    coef_volume_df.to_excel(writer, sheet_name='Volume-features', index=False)
    coef_cluster_df.to_excel(writer, sheet_name='Cluster-features', index=False)

#%% Random forest - age-groups

#Age-group 1
index1 = np.where(y < 8.6)
X_flat1 = X_flat[index1, :][0]
y1 = y[index1]
y_true1, y_pred1, coef1 = ForestRegCV_func(X_flat1, y1, n_splits = 5, param = best_param)

#Age-group 2
index2 = np.where((y >= 8.6) & (y < 17.2))
X_flat2 = X_flat[index2, :][0]
y2 = y[index2]
y_true2, y_pred2, coef2 = ForestRegCV_func(X_flat2, y2, n_splits = 5, param = best_param)

#Age-group 3
index3 = np.where(y >= 17.2)
X_flat3 = X_flat[index3, :][0]
y3 = y[index3]
y_true3, y_pred3, coef3 = ForestRegCV_func(X_flat3, y3, n_splits = 5, param = best_param)

#%% Random forest - backward selection

colnames = X_colnames + dist_colnames + vol_colnames + cluster_colnames

feature_name_ls, feature_index = feature_list_func (colnames)

result_df, plot_mat = Forest_backward_selection_func (X_flat, y, 5, best_param, feature_name_ls, feature_index, version)

#%% Random forest - HOLD OUT

# Parameter from parameter estimation:
best_param = {'bootstrap': True,        #[default = True]
              'max_depth': None,        #Max depth (length of the longest path from root to leaf) of each the decision trees in the forest [default = None]
              'max_features': 'log2',   #Maximum number of feature selected (sqrt = sqrt(total number of features)) [default = 1.0, "sqrt", "log2", 0.5]
              'max_leaf_nodes': None,   #Maximum number of leaf nodes in each tree in the forest [default = None]
              'max_samples': None,      #Maximum number of samples used in random sampling. decimal = % of total number of samples [default = None, 0.1]
              'min_impurity_decrease': 0.0, #Minimum decrease in impurity needed for a node to split [default = 0.0, 0.5]
              'min_samples_leaf': 5,    #Minimum number of samples required to be at a leaf node [default = 1, 5, 10]
              'min_samples_split': 5,   #Minimum number of samples required to split a node (=reduce impurity) [default = 2, 5, 10]
              'n_estimators': 300,      #Number of trees in the forest [default = 100, 125, 150, 175, 200, 225, 250]
              'loss_function': 'squared_error'    #absolute_error, squared_error
}

y_true_ML, y_pred_ML, y_true_HO, y_pred_HO =  ForestRegHoldout_func (X_flat, y, 5, best_param, X_holdout, y_holdout)

