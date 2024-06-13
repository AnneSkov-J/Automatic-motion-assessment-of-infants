#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: Motion Model, Clustering
Description: 
This file contain three clustering methods: K-Means, K-Means with Mahalanbis, and GMM

Created on Thu Jun 13 15:32:51 2024
@author: anne
"""

#%% Load packages from setting
import os
path_root = '...'
os.chdir(path_root)
from settings import *
from functions import * 

#%% K-MEANS CLUSTERING

age = 1
run = 0 #if 0 = iteration

print('AGE: ', str(age))
print('Iteration: ', run)


#####################
###### SETTINGS #####
#####################

#Define paths
path_Kmeans = os.path.join(path_root, '2.Clustering/Kmeans')
path_PCA = os.path.join(path_scripts, '2.Segmentation/PCA/PCA1')

## Define data
if age == 1:
    path_age = os.path.join(path_Kmeans, 'Age1')
    
    #Load data
    angles_df = pd.read_csv(os.path.join(path_agedata, 'angles1.csv'))
    path_file = os.path.join(path_PCA, 'Age1/PCA1_1_index.csv')
    with open(path_file, newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader: #there is only one row
            segment_index = row
    
    if run > 0: #if not iteration-loop
        path_file = os.path.join(path_age, 'outlier_index_' + str(run-1) + '.csv')
        with open(path_file, newline='') as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader: #there is only one row
                outlier_index_prev = row
        outlier_index_prev = np.array(outlier_index_prev, dtype=int)

        #Load cluster centers from previous iteration
        path_file = os.path.join(path_age, 'cluster_center_' + str(run-1) + '.csv')
        cluster_centers_df = pd.read_csv(path_file)
        cluster_centers_prev = cluster_centers_df.to_numpy()
       
elif age == 2:
    path_age = os.path.join(path_Kmeans, 'Age2')
    
    #Load data
    angles_df = pd.read_csv(os.path.join(path_agedata, 'angles2.csv'))
    path_file = os.path.join(path_PCA, 'Age2/PCA1_1_index.csv')
    with open(path_file, newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader: #there is only one row
            segment_index = row
           
    if run > 0: #if not iteration-loop
        #Load outliers from previous iteration
        path_file = os.path.join(path_age, 'outlier_index_' + str(run-1) + '.csv')
        with open(path_file, newline='') as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader: #there is only one row
                outlier_index_prev = row
        outlier_index_prev = np.array(outlier_index_prev, dtype=int)

        #Load cluster centers from previous iteration
        path_file = os.path.join(path_age, 'cluster_center_' + str(run-1) + '.csv')
        cluster_centers_df = pd.read_csv(path_file)
        cluster_centers_prev = cluster_centers_df.to_numpy() 
  
elif age == 3:
    path_age = os.path.join(path_Kmeans, 'Age3')
    
    #Load data
    angles_df = pd.read_csv(os.path.join(path_agedata, 'angles3.csv'))
    path_file = os.path.join(path_PCA, 'Age3/PCA1_1_index.csv')
    with open(path_file, newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader: #there is only one row
            segment_index = row
    
    if run > 0: #if not iteration-loop
        #Load outliers from previous iteration
        path_file = os.path.join(path_age, 'outlier_index_' + str(run-1) + '.csv')
        with open(path_file, newline='') as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader: #there is only one row
                outlier_index_prev = row
        outlier_index_prev = np.array(outlier_index_prev, dtype=int)

        #Load cluster centers from previous iteration
        path_file = os.path.join(path_age, 'cluster_center_' + str(run-1) + '.csv')
        cluster_centers_df = pd.read_csv(path_file)
        cluster_centers_prev = cluster_centers_df.to_numpy()

######################
###### ITERATION #####
######################

while True:
    
    print('ITERATION ', str(run))

    ####################################
    ###### SEGMENT AND INTERPOLATE #####
    ####################################
    
    #Segment and interpolate angles data
    segment_len = [int(segment_index[i]) - int(segment_index[i-1]) for i in range(1,len(segment_index))] #Calculate segment length
    percentile_99 = np.percentile(np.array(segment_len), 99) 
    # Extract and interpolate data segments
    data_segment_ls, interpol_segment_ls = segment_interpolation_func (angles_df, segment_index, 5, int(percentile_99)) #13249 segments, 13249 interpolated segments
    
    #Flatten segmented data
    X = interpol_segment_ls.copy()
    X_flat = [matrix.flatten() for matrix in X]
    X_flat = np.array(X_flat) 
    
    #Prepare data for clustering
    X_all = X_flat.copy() #The entire X data set
    
    #Remove outliers from last iteration from X inliers
    if run > 0: #if not iteration-loop
        X_in = np.delete(X_all, outlier_index_prev, axis=0)
    else:
        X_in = X_flat.copy() #X data containing only the inliers 
    
    ###################
    ###### KMEANS #####
    ###################
    num_clusters = 50
    
    print('Running K-means clustering')
    if run == 0:
        kmeans = KMeans(n_clusters = num_clusters, init='k-means++')
    elif run > 0:
        kmeans = KMeans(n_clusters = num_clusters, init=cluster_centers_prev)
    #Parameters:
        #n_clusters = number of clusters
        #init = method for initiation ['random', 'k-means++']
        #n_init = number of times to repeat initiation step [‘auto’ or int], default = if 'random' = 11, if 'kmeans++' = 1
        #max_iter = maximum iterations of algoritm
    #Fit the model to the reduced X data set (only containing the insiders)
    kmeans.fit(X_in)
    
    #Cluster the entire data set X
    cluster_labels = kmeans.predict(X_all)
    
    #Extract cluster-center locations
    cluster_centers = kmeans.cluster_centers_ #Centroids of the clusters [num_cluster, data_dim]
    
    #####################
    ###### OUTLIERS #####
    #####################
    ## Outlier detection using Mahalanobis distance
    print('Outlier detection')
    
    #Center X_all data before computing Mahalanobis distance
    X_centered = X_all - np.mean(X_all, axis = 0)
        
    #Compute covariance matrices for each cluster (using pseudoinverse)
    cov_mat_dt = {}
    #Loop through all unique clusters
    cluster_labels_unique = np.unique(cluster_labels)
    for label_i in cluster_labels_unique:
        #Extracting points assigned to the cluster
        cluster_points = X_centered[cluster_labels == label_i]
        if cluster_points.shape[0] > 1: #There must be more than one data point in the cluster to calculate the covariance matrix
        #Calcualte covariance matrix for the cluster
        #regularization = 1e-3 * np.eye(cluster_points.shape[1]) #adding a small regularization term to the diagnonal of the covariance matrix to make it invertible (it is not invertible because its singular (some diagonal elements are 0))
            cov_mat = np.cov(cluster_points, rowvar=False) #+ regularization
            inv_cov_mat = np.linalg.pinv(cov_mat) #calculate Moore-Penrose pseudoinverse of the covariance matrix
        #inv_cov_mat = np.linalg.inv(cov_mat)
        #Store inverse covariance matrix
            cov_mat_dt[label_i] = inv_cov_mat
        else:
            cov_mat_dt[label_i] = np.array([0])
            print('*')
    
    # Compute Mahalanobis distance for each data point
    mahalanobis_distances = []
    #Loop though data points and their cluster labels
    for i, label_i in enumerate(cluster_labels):
        #Extract inverse covariance matrix and cluster center for the cluster
        inv_cov_mat = cov_mat_dt[label_i]
        center_i = cluster_centers[label_i]
        if inv_cov_mat.shape[0] > 1: #if there are more than one data point in the cluster
            mahalanobis = distance.mahalanobis(X_centered[i], center_i, inv_cov_mat)
        else:
            mahalanobis = 0
        mahalanobis_distances.append(mahalanobis)
     
    #Define threshold for outlier detection
    alpha = 0.05 #significance level => 0.05 = 95% confindence
    df = 36 #number of axis
    threshold = chi2.ppf(1 - alpha, df) #using chi^2 distribution to determine the critical value
    
    #Outlier index
    print(mahalanobis_distances)
    outlier_index = np.where(mahalanobis_distances > threshold)[0] 
    
    if run > 0: #if not iteration-loop
        #If outliers detected
        if len(outlier_index) > 0:
            print('* Number of outliers detected: ', len(outlier_index), '/', X_all.shape[0])
            
            # Compare with outlier_index with last iteration 
            new_outliers = sum(1 for index in outlier_index if index not in outlier_index_prev)
            n_outlier_diff = abs(len(outlier_index_prev) - len(outlier_index))
            print('* Number of new outliers detected: ', new_outliers)
            print('* Difference between previous outliers and present outliers: ', n_outlier_diff)
        
        #If no outliers detected
        ## Break the loop = convergence
        else:
            print('* No outliers detected')
    
    ##################
    ###### SAVE ######
    ##################
    
    print('Save')
    
    path_file = os.path.join(path_age, 'cluster_center_' + str(run) + '.csv')
    cluster_centers_df = pd.DataFrame(cluster_centers)
    cluster_centers_df.to_csv(path_file, index=False) 
    
    path_file = os.path.join(path_age, 'cluster_labels_' + str(run) + '.csv')
    with open(path_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(cluster_labels)
    
    path_file = os.path.join(path_age, 'mahala_dist_' + str(run) +'.csv')
    with open(path_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(mahalanobis_distances)
    
    path_file = os.path.join(path_age, 'outlier_index_' + str(run) + '.csv')
    with open(path_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(outlier_index)
        
    ####################
    ###### UPDATE ######
    ####################
    
    print('Update')
    run += 1
    
    del X_flat, X_all, X_in, X_centered
    
    outlier_index_prev = outlier_index.copy()
    cluster_centers_prev = cluster_centers.copy()
    del cluster_centers, cluster_labels, mahalanobis_distances, outlier_index
    del cov_mat_dt, cluster_labels_unique
    
    print('ITERATION DONE')
    
#%% K-MEANS CLUSTERING WITH MAHALANOBIS

age = 1
run = 0 #if 0 = iteration

print('AGE: ', str(age))
print('Iteration: ', run)

#Define paths
path_Kmeans = os.path.join(path_scripts, '2.Clustering/KmeansMahala')
path_PCA = os.path.join(path_scripts, '2.Segmentation/PCA/PCA1')

## Define data
if age == 1:
    path_age = os.path.join(path_Kmeans, 'Age1')
    
    #Load data
    angles_df = pd.read_csv(os.path.join(path_agedata, 'angles1.csv'))
    path_file = os.path.join(path_PCA, 'Age1/PCA1_1_index.csv')
    with open(path_file, newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader: #there is only one row
            segment_index = row
    
    if run > 0: #if not iteration-loop
        #Load outliers from previous iteration
        path_file = os.path.join(path_age, 'outlier_index_' + str(run-1) + '.csv')
        with open(path_file, newline='') as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader: #there is only one row
                outlier_index_prev = row
        outlier_index_prev = np.array(outlier_index_prev, dtype=int)
       
elif age == 2:
    path_age = os.path.join(path_Kmeans, 'Age2')
    
    #Load data
    angles_df = pd.read_csv(os.path.join(path_agedata, 'angles2.csv'))
    path_file = os.path.join(path_PCA, 'Age2/PCA1_1_index.csv')
    with open(path_file, newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader: #there is only one row
            segment_index = row
           
    if run > 0: #if not iteration-loop
        #Load outliers from previous iteration
        path_file = os.path.join(path_age, 'outlier_index_' + str(run-1) + '.csv')
        with open(path_file, newline='') as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader: #there is only one row
                outlier_index_prev = row
        outlier_index_prev = np.array(outlier_index_prev, dtype=int)
  
elif age == 3:
    path_age = os.path.join(path_Kmeans, 'Age3')
    
    #Load data
    angles_df = pd.read_csv(os.path.join(path_agedata, 'angles3.csv'))
    path_file = os.path.join(path_PCA, 'Age3/PCA1_1_index.csv')
    with open(path_file, newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader: #there is only one row
            segment_index = row
    
    if run > 0: #if not iteration-loop
        #Load outliers from previous iteration
        path_file = os.path.join(path_age, 'outlier_index_' + str(run-1) + '.csv')
        with open(path_file, newline='') as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader: #there is only one row
                outlier_index_prev = row
        outlier_index_prev = np.array(outlier_index_prev, dtype=int)

####################################
###### SEGMENT AND INTERPOLATE #####
####################################

#Segment and interpolate angles data
segment_len = [int(segment_index[i]) - int(segment_index[i-1]) for i in range(1,len(segment_index))] #Calculate segment length
percentile_99 = np.percentile(np.array(segment_len), 99) 
# Extract and interpolate data segments
data_segment_ls, interpol_segment_ls = segment_interpolation_func (angles_df, segment_index, 5, int(percentile_99)) #13249 segments, 13249 interpolated segments

#Flatten segmented data
X = interpol_segment_ls.copy()
X_flat = [matrix.flatten() for matrix in X]
X_flat = np.array(X_flat) 

#Prepare data for clustering
X_all = X_flat.copy() #The entire X data set
X_in = X_flat.copy() #X data containing only the inliers 

#Remove outliers from last iteration from X inliers
if run > 0: #if not iteration-loop
    X_in = np.delete(X_all, outlier_index_prev, axis=0)

####################################
###### INITLIALISE WITH KMEANS #####
####################################
num_clusters = 50

print('Initialise clusters with K-means clustering')
kmeans = KMeans(n_clusters = num_clusters, init='k-means++')
#Parameters:
    #n_clusters = number of clusters
    #init = method for initiation ['random', 'k-means++']
    #n_init = number of times to repeat initiation step [‘auto’ or int], default = if 'random' = 11, if 'kmeans++' = 1
    #max_iter = maximum iterations of algoritm
#Fit the model to the reduced X data set (only containing the insiders)
kmeans.fit(X_in)

#Cluster the insider data set X
cluster_labels = kmeans.predict(X_in)

#Extract cluster-center locations
cluster_centers = kmeans.cluster_centers_ #Centroids of the clusters 

n_clusters = kmeans.n_clusters
n_features = X_in.shape[1]

#SAVE
path_file = os.path.join(path_age, 'cluster_labelsKmeans_' + str(run) + '.csv')
with open(path_file, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(cluster_labels)


cluster_labels_prev = cluster_labels.copy()
convergence = False

while convergence == False:

    ############################
    ###### UPDATE CLUSTERS #####
    ############################
    ##Update clusters with Mahalanobis distance
    print('Update clusters with Mahalanobis distance')
    
    #Center X_in data before computing Mahalanobis distance
    X_centered = X_in - np.mean(X_in, axis = 0)
    
    print('* Cluster_label size', cluster_labels_prev.shape, '*')
    print('* Data size', X_centered.shape, '*')
    #Compute covariance matrices for each cluster (using pseudoinverse)
    print('..compute covariance matrices')
    cov_mat_dt = {}
    cluster_labels_unique = np.unique(cluster_labels_prev)
    for label_i in cluster_labels_unique:
        #print(label_i)
        cluster_points = X_centered[cluster_labels_prev == label_i] #Extracting points assigned to the cluster
        #Calcualte covariance matrix for the cluster
        #regularization = 1e-3 * np.eye(cluster_points.shape[1]) #adding a small regularization term to the diagnonal of the covariance matrix to make it invertible (it is not invertible because its singular (some diagonal elements are 0))
        cov_mat = np.cov(cluster_points, rowvar=False) #+ regularization
        if cov_mat is not None and len(cov_mat.shape) > 0 and cov_mat.shape[0] > 0:
            #print(cov_mat.shape)
 	    inv_cov_mat = np.linalg.pinv(cov_mat) #calculate Moore-Penrose pseudoinverse of the covariance matrix
	    cov_mat_dt[label_i] = inv_cov_mat #Store inverse covariance matrix
    
    #PREDICT on all data
    #Update cluster labels and cluster centers on all data
    print('..update cluster labels on all data')
    X_centered = X_all - np.mean(X_all, axis = 0)
    cluster_labels = []
    for i in range(X_centered.shape[0]): #loop over insider data points
        mahalanobis_i = []
        for label_j in cluster_labels_unique:
            inv_cov_mat = cov_mat_dt[label_j]
            center_j = cluster_centers[label_j]
            mahalanobis = distance.mahalanobis(X_centered[i], center_j, inv_cov_mat)
            mahalanobis_i.append(mahalanobis)
        #Find the lowest Mahalanobis distance
        cluster_i = mahalanobis_i.index(np.nanmin(mahalanobis_i))
        #Find new cluster label
        cluster_labels.append(cluster_i)
    cluster_labels = np.asarray(cluster_labels) #convert from list to array
        
    #SAVE
    path_file = os.path.join(path_age, 'cluster_labelsMahala_' + str(run) + '.csv')
    with open(path_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(cluster_labels)

    #COMPARE clusters from Kmeans with Mahalanobis
    if len(cluster_labels) == len(cluster_labels_prev):
        compare = np.sum(cluster_labels == cluster_labels_prev)
        print('Number of matching cluster labels: ', compare, '/', len(cluster_labels))
        print(compare, 0.5*len(cluster_labels))
        
        if compare >= 0.5*len(cluster_labels):
            convergence = True
            print('Convergence...')
    
    #Find new cluster center
    print('..update cluster centers')
    cluster_centers = np.zeros((n_clusters, n_features))
    for i in range(n_clusters):
        #cluster_points = X_centered[cluster_labels == label_i]
        cluster_i = [label == i for label in cluster_labels] #bool index
        cluster_points = X_centered[cluster_i]
        #print(X_centered.shape, cluster_points.shape)
        if cluster_points.shape[0] != 0:
            cluster_centers[i, :] = np.nanmean(cluster_points, axis = 0)
    
    #SAVE
    path_file = os.path.join(path_age, 'cluster_center_' + str(run) + '.txt')
    np.savetxt(path_file, cluster_centers)
    
    #Save cluster labels from Mahalanobis-update as previous cluster labels for new iteration
    cluster_labels_prev = cluster_labels.copy()
 
#####################
###### OUTLIERS #####
#####################

## Outlier detection using Mahalanobis distance
print('Outlier detection')

#Center X_all data before computing Mahalanobis distance
X_centered = X_all - np.mean(X_all, axis = 0)
    
#Compute covariance matrices for each cluster (using pseudoinverse)
cov_mat_dt = {}
#Loop through all unique clusters
cluster_labels_unique = np.unique(cluster_labels)
for label_i in cluster_labels_unique:
    #print(label_i)
    #Extracting points assigned to the cluster
    cluster_points = X_centered[cluster_labels == label_i]
    #Calcualte covariance matrix for the cluster
    #regularization = 1e-3 * np.eye(cluster_points.shape[1]) #adding a small regularization term to the diagnonal of the covariance matrix to make it invertible (it is not invertible because its singular (some diagonal elements are 0))
    cov_mat = np.cov(cluster_points, rowvar=False) #+ regularization
    inv_cov_mat = np.linalg.pinv(cov_mat) #calculate Moore-Penrose pseudoinverse of the covariance matrix
    #inv_cov_mat = np.linalg.inv(cov_mat)
    #Store inverse covariance matrix
    cov_mat_dt[label_i] = inv_cov_mat

# Compute Mahalanobis distance for each data point
mahalanobis_distances = []
#Loop though data points and their cluster labels
for i, label_i in enumerate(cluster_labels):
    #Extract inverse covariance matrix and cluster center for the cluster
    inv_cov_mat = cov_mat_dt[label_i]
    center_i = cluster_centers[label_i]
    mahalanobis = distance.mahalanobis(X_centered[i], center_i, inv_cov_mat)
    mahalanobis_distances.append(mahalanobis)
 
#Define threshold for outlier detection
alpha = 0.05 #significance level => 0.05 = 95% confindence
df = 36 #number of axis
threshold = chi2.ppf(1 - alpha, df) #using chi^2 distribution to determine the critical value

#Outlier index
outlier_index = np.where(mahalanobis_distances > threshold)[0] 

if run > 0: #if not iteration-loop
    #If outliers detected
    if len(outlier_index) > 0:
        print('* Number of outliers detected: ', len(outlier_index), '/', X_all.shape[0])
        
        # Compare with outlier_index with last iteration 
        new_outliers = sum(1 for index in outlier_index if index not in outlier_index_prev)
        n_outlier_diff = abs(len(outlier_index_prev) - len(outlier_index))
        print('* Number of new outliers detected: ', new_outliers)
        print('* Difference between previous outliers and present outliers: ', n_outlier_diff)
    
    #If no outliers detected
    ## Break the loop = convergence
    else:
        print('* No outliers detected')

##################
###### SAVE ######
##################

print('Save')

path_file = os.path.join(path_age, 'cluster_center_' + str(run) + '.csv')
cluster_centers_df = pd.DataFrame(cluster_centers)
cluster_centers_df.to_csv(path_file, index=False) 

path_file = os.path.join(path_age, 'cluster_labels_' + str(run) + '.csv')
with open(path_file, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(cluster_labels)

path_file = os.path.join(path_age, 'mahala_dist_' + str(run) +'.csv')
with open(path_file, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(mahalanobis_distances)

path_file = os.path.join(path_age, 'outlier_index_' + str(run) + '.csv')
with open(path_file, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(outlier_index)

print('ITERATION DONE')

#%% GAUSSIAN MIXTURE MODEL

age = 1
run = 0 #if 0 = iteration

print('AGE: ', str(age))
print('Iteration: ', run)

#Define paths
path_GMM = os.path.join(path_scripts, '2.Clustering/GMM')
path_PCA = os.path.join(path_scripts, '2.Segmentation/PCA/PCA1')

## Define data
if age == 1:
    path_age = os.path.join(path_GMM, 'Age1')
    
    #Load data
    angles_df = pd.read_csv(os.path.join(path_agedata, 'angles1.csv'))
    path_file = os.path.join(path_PCA, 'Age1/PCA1_1_index.csv')
    with open(path_file, newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader: #there is only one row
            segment_index = row
    
    if run > 0: #if not iteration-loop
        #Load outliers from previous iteration
        path_file = os.path.join(path_age, 'outlier_index_' + str(run-1) + '.csv')
        with open(path_file, newline='') as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader: #there is only one row
                outlier_index_prev = row
        outlier_index_prev = np.where(outlier_index_prev)[0]
        outlier_index_prev = np.array(outlier_index_prev, dtype=int)
       
elif age == 2:
    path_age = os.path.join(path_GMM, 'Age2')
    
    #Load data
    angles_df = pd.read_csv(os.path.join(path_agedata, 'angles2.csv'))
    path_file = os.path.join(path_PCA, 'Age2/PCA1_1_index.csv')
    with open(path_file, newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader: #there is only one row
            segment_index = row
    
    if run > 0: #if not iteration-loop
        #Load outliers from previous iteration
        path_file = os.path.join(path_age, 'outlier_index_' + str(run-1) + '.csv')
        with open(path_file, newline='') as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader: #there is only one row
                outlier_index_prev = row
        outlier_index_prev = np.array(outlier_index_prev, dtype=int)
  
elif age == 3:
    path_age = os.path.join(path_GMM, 'Age3')
    
    #Load data
    angles_df = pd.read_csv(os.path.join(path_agedata, 'angles3.csv'))
    path_file = os.path.join(path_PCA, 'Age3/PCA1_1_index.csv')
    with open(path_file, newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader: #there is only one row
            segment_index = row
    
    if run > 0: #if not iteration-loop
        #Load outliers from previous iteration
        path_file = os.path.join(path_age, 'outlier_index_' + str(run-1) + '.csv')
        with open(path_file, newline='') as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader: #there is only one row
                outlier_index_prev = row
        outlier_index_prev = np.array(outlier_index_prev, dtype=int)

######################
###### ITERATION #####
######################

while True:
    
    print('ITERATION ', str(run))
    
    ####################################
    ###### SEGMENT AND INTERPOLATE #####
    ####################################
    
    #Segment and interpolate angles data
    segment_len = [int(segment_index[i]) - int(segment_index[i-1]) for i in range(1,len(segment_index))] #Calculate segment length
    percentile_99 = np.percentile(np.array(segment_len), 99) 
    # Extract and interpolate data segments
    data_segment_ls, interpol_segment_ls = segment_interpolation_func (angles_df, segment_index, 5, int(percentile_99)) #13249 segments, 13249 interpolated segments
    
    #Flatten segmented data
    X = interpol_segment_ls.copy()
    X_flat = [matrix.flatten() for matrix in X]
    X_flat = np.array(X_flat) 
    
    #Prepare data for clustering
    X_all = X_flat.copy() #The entire X data set
    
    #Remove outliers from last iteration from X inliers
    if run > 0: #if not iteration-loop
        X_in = np.delete(X_all, outlier_index_prev, axis=0)
    else: 
        X_in = X_flat.copy() #X data containing only the inliers 
    
    
    ################
    ###### GMM #####
    ################
    num_clusters = 50
    
    print('Running Gaussian Mixture Model clustering')
    
    # Initialize and fit GMM
    gmm = GaussianMixture(n_components = num_clusters, covariance_type = 'full', init_params = 'random', verbose = 2)
    gmm.fit(X_in)
    
    #Extract model parameters
    gmm_mean = gmm.means_
    gmm_cov = gmm.covariances_
    
    #Cluster the entire data set X
    cluster_labels = gmm.predict(X_all)
    
     
    #####################
    ###### OUTLIERS #####
    #####################   
    # Calculate the likelihood of each data point under the GMM
    likelihoods = gmm.score_samples(X_all)
    
    # Set a threshold for outlier detection (e.g., 5th percentile)
    threshold = np.percentile(likelihoods, 5)
    
    # Identify outliers based on the threshold
    outlier_index = [likelihoods < threshold]
    outlier_index = np.where(outlier_index)[0]
    
    if run > 0: #if not iteration-loop
        #If outliers detected
        if len(outlier_index) > 0:
            print('* Number of outliers detected: ', len(outlier_index), '/', X_all.shape[0])
            
            # Compare with outlier_index with last iteration 
            new_outliers = sum(1 for index in outlier_index if index not in outlier_index_prev)
            n_outlier_diff = abs(len(outlier_index_prev) - len(outlier_index))
            print('* Number of new outliers detected: ', new_outliers)
            print('* Difference between previous outliers and present outliers: ', n_outlier_diff)
        
        #If no outliers detected
        ## Break the loop = convergence
        else:
            print('* No outliers detected')
    
    
    ##################
    ###### SAVE ######
    ##################
    
    print('Save')
    
    path_file = os.path.join(path_age, 'gmm_mean_' + str(run) + '.csv')
    with open(path_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(gmm_mean)
    
    path_file = os.path.join(path_age, 'cluster_labels_' + str(run) + '.csv')
    with open(path_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(cluster_labels)

    path_file = os.path.join(path_age, 'likelihood_' + str(run) + '.csv')
    with open(path_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(likelihood)
    
    path_file = os.path.join(path_age, 'outlier_index_' + str(run) + '.csv')
    with open(path_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(outlier_index)
    
    ####################
    ###### UPDATE ######
    ####################
    
    print('Update')
    run += 1
    
    del X_flat, X_all, X_in
    
    outlier_index_prev = outlier_index.copy()
    del gmm_mean, cluster_labels, outlier_index
    del gmm_cov
    
    print('ITERATION DONE')
