#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Title: Motion Model, Segmentation
Description: 
This file contain the four segmentation methods: frequent pose, zero velocity, PCA and PPCA

Created on Wed Apr 10 17:48:56 2024
@author: anne
"""


#%% Load packages from setting
import os
path_root = '...'
os.chdir(path_root)
from settings import *
from functions import *

#%% FREQUENT POSE

for age in [1,2,3]:
    
    print('Age: ', str(age))
    
    print('Loading')
    
    #Define paths
    path_posedata = os.path.join(path_root, '2.Segmentation/Pose'))
    
    #Define data 
    if age == 1:
        angles_df = pd.read_csv(os.path.join(path_agedata, 'angles1.csv'))
        position_df = pd.read_csv(os.path.join(path_agedata, 'position1.csv'))
    elif age == 2:
        angles_df = pd.read_csv(os.path.join(path_agedata, 'angles2.csv'))
        position_df = pd.read_csv(os.path.join(path_agedata, 'position2.csv'))
    elif age == 3:
        angles_df = pd.read_csv(os.path.join(path_agedata, 'angles3.csv'))
        position_df = pd.read_csv(os.path.join(path_agedata, 'position3.csv'))
    
    #Construct X - a list of arrays, where each array is a frame in angles_df
    X = []  #list of data arrays
    for i in range(angles_df.shape[0]):
        X.append(angles_df.iloc[i].values)
    
    #Run model
    print('Run model')
    for max_distance in [0.1, 0.2]:
        path_posedata1 = os.path.join(path_posedata, 'max_dist_' + str(max_distance))
        
        info_dt = {}
        info_dt['age'] = [age]
        
        ## Density-Based Spatial Clustering of Applications with Noise (adjust cluster-size)
        #Define model parameters 
        #max_distance = 0.1  #The maximum distance (diameter) between points in the same cluster
        min_samples = 100   #The minimum number of points in a cluster
        info_dt['max_distance'] = [max_distance]
        info_dt['min_samples'] = [min_samples]
           
        #Run the model
        dbscan = DBSCAN(eps = max_distance, min_samples = min_samples)
        dbscan.fit(X)
    
        #Access the cluster labels (-1 indicates noise/outliers)
        cluster_labels = dbscan.labels_
        
        num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0) #The number of clusters (excluding noise)
        unclustered_indices = [i for i, label in enumerate(cluster_labels) if label == -1]
        
        cluster_count = Counter(cluster_labels)
        
        info_dt['num_clusters'] = [num_clusters]
        info_dt['cluster_labels'] = [cluster_labels]
        #info_dt['index'] = [index]
                
        #Save cluster labels 
        path_file = os.path.join(path_posedata1, 'Labels_age' + str(age) + '.csv')     
        with open(path_file, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(cluster_labels)

        #Save information directory
        path_file = os.path.join(path_posedata1, 'info' + str(age) + '_dt.csv') 
        with open(path_file, 'wb') as pickle_file:
            pickle.dump(info_dt, pickle_file)

    print('AGE DONE')
    
print('FINALLY DONE')

#%% ZERO VELOCIRY

for age in [1,2,3]:
    
    print('AGE: ' + str(age))
    
    backwards = False
    
    print('Loading')
  
    #Define paths
    path_ZVC = os.path.join(path_root, '2.Segmentation/ZVC/ZVC')
    
    ## Define data and output path
    if age == 1:
        angles_df = pd.read_csv(os.path.join(path_agedata, 'angles1.csv')) 
        TimeStamp = np.loadtxt(os.path.join(path_agedata, 'TimeStamp1.csv'), delimiter=',')
        new_infant_index = np.loadtxt(os.path.join(path_agedata, 'new_infant_index1.csv'), delimiter=',')
        path_age = os.path.join(path_ZVC, 'Age1')
        #t = 3
    elif age == 2:
        angles_df = pd.read_csv(os.path.join(path_agedata, 'angles2.csv')) 
        TimeStamp = np.loadtxt(os.path.join(path_agedata, 'TimeStamp2.csv'), delimiter=',')
        new_infant_index = np.loadtxt(os.path.join(path_agedata, 'new_infant_index2.csv'), delimiter=',')
        path_age = os.path.join(path_ZVC, 'Age2')
        #t = 4
    elif age == 3:
        angles_df = pd.read_csv(os.path.join(path_agedata, 'angles3.csv')) 
        TimeStamp = np.loadtxt(os.path.join(path_agedata, 'TimeStamp3.csv'), delimiter=',')
        new_infant_index = np.loadtxt(os.path.join(path_agedata, 'new_infant_index3.csv'), delimiter=',')
        path_age = os.path.join(path_ZVC, 'Age3') 
        #t = 5
        
    #Opposite data
    if backwards == True:
        print('Backwards')
        path_ZVC = os.path.join(path_root, '4.Segmentation/ZVC/ZVC-1')
        angles_df = angles_df.iloc[::-1] 
        
    #Directory for storing information:
    info_dt = {}
    info_dt['age'] = [age]
    
    ##Calculating angular velocity ~ frames
    print('Calculating angular velocity')
    
    #Define variables and parameters
    n_timestep = angles_df.shape[0] #number of frames in the data
    n_frame = int(n_timestep)
    frame = 1 #width
    var_ls = angles_df.columns
    
    #Calculate angular velocity for each angle in each joint
    #(the ZVC must be associated to only one DOF [Jerkins])
    var_ls = angles_df.columns
    velocity_age  = pd.DataFrame(columns = var_ls)
    
    for var in var_ls:  
        
        #Extract data
        data = angles_df[var]
        
        #Initialise
        velocity_frame = np.full(n_frame, np.nan)
    
        #Make an index of frame-positions in time
        frame_index = [i for i in range(1, n_timestep, 1)]
        
        #Calculate the movement-features for each frame = x timestamps 
        for i in range(len(frame_index)):
            frame_i = frame_index[i]      #the position in data of the i-th frame
            
            #Calculate the "distance travelled" between the i-1 and the i'th frame
            delta_dist = abs((data.iloc[frame_i] - data.iloc[frame_i - frame]))
            
            #Calculate the time-difference between the i-1 and i'th frame
            delta_t = abs(TimeStamp[frame_i] - TimeStamp[frame_i - frame])
            
            #Calculate angular velocity between the i-1 and the i'th frame
            velocity_frame[i] = delta_dist/delta_t
            
        #Store
        velocity_age[var] = velocity_frame
    
    ##############################
    ##### Jerkins version 1 ######
    ##############################
    
    #Exclude locked angles
    zero_columns = velocity_age.columns[velocity_age.eq(0).all()]
    velocity_age.drop(columns=zero_columns, inplace=True)
    var_ls = velocity_age.columns
    
    #Initialise
    ZVC_df = pd.DataFrame(columns = var_ls)
    
    ##Label potential ZVC
    print('Label potential ZVCs')
    for var in var_ls:
        data = velocity_age[var] #1 DF = 1 axis
        #data = data.dropna()     #The last data-point is nan
        
        ##1. Label all potential ZVC's in data-variable
        ZVC_index = [data == 0][0] 
        
        #Save
        ZVC_df[var] = ZVC_index
    
    
    ##Label final ZVCs
    print('Label final ZVCs')
    #ZVC_df = pd.read_csv(os.path.join(path_ZVC, 'ZVC_jerkins1_age2.csv'), index_col = 0) 
    
    #Group extremities of the body
    RArm = ['RUArmAngle1','RUArmAngle2', 'RUArmAngle3','RLArmAngle1', 'RLArmAngle2'] #'RLArmAngle3'
    LArm = ['LUArmAngle1', 'LUArmAngle2', 'LUArmAngle3','LLArmAngle1', 'LLArmAngle2'] #'LLArmAngle3'
    RLeg = ['RULegAngle1', 'RULegAngle2', 'RULegAngle3', 'RLLegAngle1', 'RLLegAngle2', 'RFootAngle1','RFootAngle3'] #'RLLegAngle3' 'RFootAngle2'
    LLeg = ['LULegAngle1', 'LULegAngle2', 'LULegAngle3', 'LLLegAngle1', 'LLLegAngle2', 'LFootAngle1', 'LFootAngle3'] #'LLLegAngle3' 'LFootAngle2'
    
    #Define parameters
    t = 4 #looking at t rows at a time (default: t=1) [t is defined in loop]
    tau = 3 #number of frames between cuts [tau is defined in loop]
    
    info_dt['t'] = [t]
    info_dt['tau'] = [tau]
    
    #Initliase ZVC-index
    ZVC1_bool = [False for i in range(t)]
    
    #Extract segments 
    for i in range(t, ZVC_df.shape[0]): #index = timestamps
        
        #Extract the i-t : i rows from ZVC_df
        row_t = ZVC_df.iloc[i : i+t] 
        
        #Count the number of potential ZVC-labels in each extremity in t time steps before i (including row i)
        count_RArm_t = np.sum(np.sum(row_t[RArm], axis = 0), axis = 0)
        count_LArm_t = np.sum(np.sum(row_t[LArm], axis = 0), axis = 0)
        count_RLeg_t = np.sum(np.sum(row_t[RLeg], axis = 0), axis = 0)
        count_LLeg_t = np.sum(np.sum(row_t[LLeg], axis = 0), axis = 0)
        
        #If the last tau rows has no final ZVC
        if not any(ZVC1_bool[-tau:]):
            
            #If one extremity has ZVC-labels in all angles in all t time steps = final ZVC label
            if count_RArm_t == len(RArm)*t or count_LArm_t == len(LArm)*t or count_RLeg_t == len(RLeg)*t or count_LLeg_t == len(LLeg)*t:
                ZVC1_bool.append(True)
            #Else no final ZVC label
            else:
                ZVC1_bool.append(False)
        #Else no final ZVC label
        else:
            ZVC1_bool.append(False)

    print('Number of ZVCs', np.sum(ZVC1_bool),'/', len(ZVC1_bool))
    info_dt['n_cuts'] = [np.sum(ZVC1_bool)]
        
    #Compare with new_infant index = to chech if the segmentation identify a change in infants    
    frame_index = [i for i in range(len(TimeStamp))] #first element is removed since ang vel is nan
    frame_index = np.array(frame_index)
    ZVC1_index = frame_index[ZVC1_bool]
    compare = [i for i in ZVC1_index if i in new_infant_index] 
    print('Age ', str(age), 'identified ', str(len(compare)), ' infant-changes')
    
    #Store in information-directory
    info_dt['n_new_infants'] = len(compare)
    info_dt['New_infant_index'] = compare
        
    #Save to file
    print('Save')
    ZVC_df.to_csv(os.path.join(path_age, 'ZVC_df.csv'), index=True)
    
    path_file = os.path.join(path_age, 'ZVC_index.csv')
    with open(path_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(ZVC1_index)
    
    path_file = os.path.join(path_age, 'Info_dt.pkl')
    with open(path_file, 'wb') as pickle_file:
        pickle.dump(info_dt, pickle_file)

    print('AGE DONE')
    
    #Clear before running next age
    locals().clear()
    
print('FINALLY DONE')

#%% PRINCIPAL COMPONENT ANALYSIS - INIT

for age in [1,2,3]:
        
    print('AGE: ', str(age))
    
    #Define paths
    path_PCA = os.path.join(path_root, '2.Segmentation/PCA/PCA1')
    
    ## Define data
    if age == 1:
        #position_df = pd.read_csv(os.path.join(path_agedata, 'position1.csv')) 
        angles_df = pd.read_csv(os.path.join(path_agedata, 'angles1.csv'))
        #TimeStamp = np.loadtxt(os.path.join(path_agedata, 'TimeStamp1.csv'), delimiter=',')
        new_infant_index = np.loadtxt(os.path.join(path_agedata, 'new_infant_index1.csv'), delimiter=',')
        path_age = os.path.join(path_PCA, 'Age1')
    elif age == 2:
        #position_df = pd.read_csv(os.path.join(path_agedata, 'position2.csv')) 
        angles_df = pd.read_csv(os.path.join(path_agedata, 'angles2.csv'))
        #TimeStamp = np.loadtxt(os.path.join(path_agedata, 'TimeStamp2.csv'), delimiter=',')
        new_infant_index = np.loadtxt(os.path.join(path_agedata, 'new_infant_index2.csv'), delimiter=',')
        path_age = os.path.join(path_PCA, 'Age2')
    elif age == 3:
        #position_df = pd.read_csv(os.path.join(path_agedata, 'position3.csv')) 
        angles_df = pd.read_csv(os.path.join(path_agedata, 'angles3.csv'))
        #TimeStamp = np.loadtxt(os.path.join(path_agedata, 'TimeStamp3.csv'), delimiter=',')
        new_infant_index = np.loadtxt(os.path.join(path_agedata, 'new_infant_index3.csv'), delimiter=',')
        path_age = os.path.join(path_PCA, 'Age3')
    
    #Opposite data
    #angles_df = angles_df[::-1]
    
    #Dataframe for storing information and plots:
    dimension_df = pd.DataFrame()
    plot_dt = {}
    
    #Center of motion
    ## Mean over frames
    center = angles_df.mean()
    
    #Center the frames (zero-mean data)
    ## frame_center = frame - center
    angles_center_df = angles_df - center
    del center, angles_df
    print('Angles Dimensions: ', angles_center_df.shape)
    
    #SVD decomposition
    # D = USV^T
    #U, S, Vt = np.linalg.svd(angles_center_df)
    #V = Vt.T
    #U = orthonomal matrix = orthonormal vectors u
    #S = Sigma = diagonal matrix = singular values (eigenvalue) in diagonal = how much variability each PC descirbe
    #V = orthonomal matrix = orthonormal vectors v = PC's (eigenvectors)
    
    ##Use PCA for SVD decomposition (to get rid of the large U-matrix)
    #Compute the covariance matrix from the centered data
    cov_mat = np.cov(angles_center_df, rowvar = False)
    
    #Compute eigenvalues and eigenvectors from covariance matrix
    eigen_val, eigen_vec = np.linalg.eigh(cov_mat) #using the sample covariance = need to scale eigen values to become singular calues
   
    #Sort eigenvalues and -vectors according to descending eigenvalues
    sort_index = np.argsort(eigen_val)[::-1]
    Vt = eigen_vec[:, sort_index]
    V = Vt.T
    num_samples = angles_center_df.shape[0]
    print('*num_samples: ', num_samples)
    print('*eigen_val: ', eigen_val)
    S = np.sqrt(eigen_val[sort_index] * num_samples) #Sort and scale eigenvalues to become singular values (scale by sqr and num of samples)
    
    print('S dim: ', S.shape)
    print('V dim: ', V.shape)
      
    #Save PCA result
    path_file = os.path.join(path_age, 'V' + str(age) + '.csv')
    np.savetxt(path_file, V, delimiter = ',')
    
    path_file = os.path.join(path_age, 'S' + str(age) + '.csv')
    np.savetxt(path_file, S, delimiter = ',')
    
    #Plot variance explained:
    S1 = S[~np.isnan(S)]
    explained_variance = (S1 ** 2) / np.sum(S1 ** 2)
    cumulative_variance = np.cumsum(explained_variance) / np.sum(explained_variance)
    plot_dt['exp_var'] = [explained_variance]
    plot_dt['cum_var'] = [cumulative_variance]
    
    plot_dt['explained_variance'] = explained_variance
    plot_dt ['cumulative_variance'] = cumulative_variance
    fig = plt.figure(figsize = (12, 9))
    plt.grid(axis = 'y', alpha = 0.75)
    plt.title('Variance explained', fontsize = big)
    plt.plot(range(len(cumulative_variance)), cumulative_variance, color = DTU_color[0])
    plt.axhline(y=0.99, linestyle = '--', color = DTU_color[1])
    plt.xlabel('Principal components')
    plt.ylabel('Explained variance')
    plt.show()
    
    path_file = os.path.join(path_age, 'Variance_explained' + str(age) + '.png')
    plt.savefig(path_file)
    del S1
    
    #Define dimensionality needed to describe motion in each frame
    ## Er > threshold < 1
    threshold = 0.9
    k = 30 #1 seconds
    r_ls = []
    e_mean_ls = []
    Er_mean_ls = []
    e_df = pd.DataFrame()
    Er_df = pd.DataFrame()
    
    #Remove potential nan from S
    S = S[~np.isnan(S)]
    
    #Looping over dimensionalities
    for r in range(1, angles_center_df.shape[1], 1):
        print('r: ' ,r)
        #Select the first r PC's (eigenvectors) = the basis of the hyperplane
        Vr = V[:,:r]
        
        e_ls = []
        Er_ls = []    
        for i in range(angles_center_df.shape[0]):
                   
            frame = angles_center_df.iloc[i:i+k]
            
            #Project all frames on the "optimal" hyperplane
            frame_project = frame @ Vr
            #Reconstruct data using lower-dimensional representation
            frame_recon = np.dot(frame_project, np.transpose(Vr))
            
            #Calculate projection error
            e = np.linalg.norm((frame - frame_recon).values)
            
            #Calculate error ratio
            Er = np.sum(S[:r]**2) / np.sum(S**2)
            
            #Store
            e_ls.append(e)
            Er_ls.append(Er)
            
            #Clear
            del frame, frame_project, frame_recon, e, Er
        
        r_ls.append(r)
        e_mean_ls.append(np.mean(e_ls))
        Er_mean_ls.append(np.mean(Er_ls))
        print(np.mean(Er_ls))
        e_df[r] = e_ls
        Er_df[r] = Er_ls
            
        if np.mean(Er_ls) >= threshold:
            print('Dimensionality: ', r)
            break
        
        #Clean
        del Vr
    
    #Store in information-directory
    dimension_df['r_ls'] = r_ls
    dimension_df['e_ls'] = e_mean_ls
    dimension_df['Er_ls'] = Er_mean_ls

    path_file = os.path.join(path_age, 'Dimension_r' + str(age) + '.csv')
    dimension_df.to_csv(path_file, index = False)
          
    print('AGE DONE')
    
print('FINALLY DONE')

#%% PRINCIPAL COMPONENT ANALYSIS

for age in [1,2,3]:
    
    print('AGE: ', str(age))

    backwards = False #True
    
    print('Load settings')
    
    #Define paths
    path_PCA = os.path.join(path_root, '2.Segmentation/PCA/PCA1')
    
    # Define data
    if age == 1:
        r = 16 #Dimensionality defined in PCA_init
        angles_df = pd.read_csv(os.path.join(path_agedata, 'angles1.csv'))
        new_infant_index = np.loadtxt(os.path.join(path_agedata, 'new_infant_index1.csv'), delimiter=',')
        path_age = os.path.join(path_PCA, 'Age1')
    elif age == 2:
        r = 16 #Dimensionality defined in PCA_init
        angles_df = pd.read_csv(os.path.join(path_agedata, 'angles2.csv'))
        new_infant_index = np.loadtxt(os.path.join(path_agedata, 'new_infant_index2.csv'), delimiter=',')
        path_age = os.path.join(path_PCA, 'Age2')
    elif age == 3:
        r = 16 #Dimensionality defined in PCA_init
        angles_df = pd.read_csv(os.path.join(path_agedata, 'angles3.csv'))
        new_infant_index = np.loadtxt(os.path.join(path_agedata, 'new_infant_index3.csv'), delimiter=',')
        path_age = os.path.join(path_PCA, 'Age3')  

    #Opposite data
    if backwards == True:
        print('Backwards')
        path_PCA = os.path.join(path_root, '4.Segmentation/PCA/PCA1-1')
        angles_df = angles_df.iloc[::-1] 

    #Center of motion
    ## Mean over frames
    center = angles_df.mean()
    
    #Center the frames (zero-mean data)
    ## frame_center = frame - center
    angles_center_df = angles_df - center
    del center, angles_df
    

    ####################################
    ###### METHOD 1 (Adaptive STD) #####
    ####################################
    
    print('METHOD 1 (adaptive std)')
    
    #Directory for storing information and plot data:
    info_dt = {}
    info_dt['age'] = [age]
    info_dt['r'] = [r]
    
    ## Segmentation
    #Initialise 
    n_frame = angles_center_df.shape[0]
    
    k = 30 #frames = 1 sec
    l = 5
    info_dt['k'] = [k]
    info_dt['l'] = [l]
    CUT_i = 0
    
    e_vec = np.full(n_frame, np.nan)
    d_vec = np.full(n_frame, np.nan)
    d_mean_vec = np.full(n_frame, np.nan)
    d_std_vec = np.full(n_frame, np.nan)
    
    cut_index = []        
    
    for i in range(1,n_frame): 
        
        #Extract data from last cut to i
        frame_df = angles_center_df.iloc[CUT_i : i+1]
        
        #SVD decomposition
        V, S = PCA_func (frame_df)
        Vr = V[:,:r]
        
        #Project frames on the hyperplane
        frame_project = frame_df @ Vr
        #Reconstruct data using lower-dimensional representation
        frame_recon = np.dot(frame_project, np.transpose(Vr))
        
        #Calculate projection error
        e_i = np.linalg.norm((frame_df - frame_recon).values)
        e_vec[i] = e_i
        
        #Initial loops
        if i > l:
            d_i = e_vec[i] - e_vec[i-l] #avoid noise
            d_vec[i] = d_i
        
        #Segmentation
        if i > k+l and i > CUT_i + l: #First cut only after k+l, there must be at least l datapoints between cuts
            
            #Calculate statistics on d
            d_prev = d_vec[CUT_i+2 : i] #+2 to not include large values around movement change
            d_prev = d_prev[~np.isnan(d_prev)]
            d_mean = np.nanmean(d_prev)
            d_std = np.nanstd(d_prev)
            d_mean_vec[i] = d_mean
            d_std_vec[i] = d_std
            
            #Determine if cut
            if d_i > d_mean + 3*d_std or d_i < d_mean - 3*d_std:
                #print('CUT: ',i )
                cut_index.append(i)
                CUT_i = i
        

    print('Evaluate and save')
    
    #Statistics on d
    info_dt['d_mean_vec'] = [d_mean_vec]
    info_dt['d_std_vec'] = [d_std_vec]
    info_dt['d_vec'] = [d_vec]
    info_dt['e_vec'] = [e_vec]
    
    #Number of cuts performed
    print('Number of CUTs: ', len(cut_index),'/', n_frame)
    info_dt['n_cuts'] = [len(cut_index)]
    info_dt['cuts_index'] = [cut_index]
    
    #Compare with new_infant index = to chech if the segmentation identify a change in infants    
    compare = [i for i in cut_index if i in new_infant_index] 
    print('Age ', str(age), 'identified ', str(len(compare)), ' infant-changes')
    
    #Store in information-directory
    info_dt['n_new_infants'] = [len(compare)]
    info_dt['New_infant_index'] = [compare]
    
    #Save segmentation index
    path_file = os.path.join(path_age, 'PCA1_1_index.csv')
    with open(path_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(cut_index)
        
    #Save information directory
    path_file = os.path.join(path_age, 'Info1_dt.pkl')
    with open(path_file, 'wb') as pickle_file:
        pickle.dump(info_dt, pickle_file)
 
    #################################
    ##### METHOD 2 (Static STD) #####
    #################################
    
    print('METHOD 2 (static std)')
    
    #Directory for storing information and plot data:
    info_dt = {}
    info_dt['age'] = [age]
    info_dt['r'] = [r]
    
    ## Segmentation
    CUT_i = 0
    cut_index = []    
    d_mean = np.mean(d_mean_vec[~np.isnan(d_mean_vec)]) #removing potential nan's
    d_std = np.std(d_mean_vec[~np.isnan(d_mean_vec)])
    info_dt['d_mean'] = [d_mean]
    info_dt['d_std'] = [d_std]
        
    for i in range(1,n_frame):
        d_i = d_vec[i]
        
        #Segmentation
        if i > k+l and i > CUT_i + l: #First cut only after k+l, there must be at least l datapoints between cuts
            
            #Calculate statistics on d
            d_prev = d_vec[CUT_i+2 : i] #+5 to not include large values around movement change
            d_prev = d_prev[~np.isnan(d_prev)]
            
            #Determine if cut
            if d_i > d_mean + 3*d_std or d_i < d_mean - 3*d_std:
                #print('CUT: ',i )
                cut_index.append(i)
                CUT_i = i
            
            del d_prev
        
    print('Evaluate and save')
        
    #Number of cuts performed
    print('Number of CUTs: ', len(cut_index),'/', n_frame)
    info_dt['n_cuts'] = [len(cut_index)]
    info_dt['cuts_index'] = [cut_index]
    
    #Compare with new_infant index = to chech if the segmentation identify a change in infants    
    compare = [i for i in cut_index if i in new_infant_index] 
    print('Age ', str(age), 'identified ', str(len(compare)), ' infant-changes')
    
    #Store in information-directory
    info_dt['n_new_infants'] = [len(compare)]
    info_dt['New_infant_index'] = [compare]
    
    #Save segmentation index
    path_file = os.path.join(path_age, 'PCA1_2_index.csv')
    with open(path_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(cut_index)
        
    #Save information directory
    path_file = os.path.join(path_age, 'Info2_dt.pkl')
    with open(path_file, 'wb') as pickle_file:
        pickle.dump(info_dt, pickle_file)
    
    print('AGE DONE')
    
print('FINALLY DONE')

#%% PROBABILISTIC PRINICPAL COMPONENT ANALYSIS

for age in [1,2,3]:
    threshold = 2000
    
    print('AGE: ', str(age))
    
    backwards = False #False
    
    print('Loading settings')

    #Define paths
    path_PCA = os.path.join(path_root, '2.Segmentation/PPCA/PPCA')
                            
    ## Define data
    if age == 1:
        path_age = os.path.join(path_PCA, 'Age1')
        
        #Dimensionality defined in PCA_init 
        r = 16
          
        #Load data
        angles_df = pd.read_csv(os.path.join(path_agedata, 'angles1.csv'))
        new_infant_index = np.loadtxt(os.path.join(path_agedata, 'new_infant_index1.csv'), delimiter=',')
   
    elif age == 2:
        path_age = os.path.join(path_PCA, 'Age2')
        
        #Dimensionality defined in PCA_init
        r = 16
           
        #Load data
        angles_df = pd.read_csv(os.path.join(path_agedata, 'angles2.csv'))
        new_infant_index = np.loadtxt(os.path.join(path_agedata, 'new_infant_index2.csv'), delimiter=',')
   
    elif age == 3:
        path_age = os.path.join(path_PCA, 'Age3')
        
        #Dimensionality defined in PCA_init 
        r = 16
        
        #Load data
        angles_df = pd.read_csv(os.path.join(path_agedata, 'angles3.csv'))
        new_infant_index = np.loadtxt(os.path.join(path_agedata, 'new_infant_index3.csv'), delimiter=',')

    #Opposite data
    if backwards == True:
        print('Backwards')
        path_PCA = os.path.join(path_root, '4.Segmentation/PPCA/PPCA-1')
        angles_df = angles_df.iloc[::-1]  

    #Directory for storing information:
    info_dt = {}
    info_dt['age'] = [age]
    info_dt['r'] = [r]
    info_dt['opposite'] = [True]
    
    #Center of motion
    ## Mean over frames
    center = angles_df.mean()
    
    #Center the frames (zero-mean data)
    angles_center_df = angles_df - center
    del center, angles_df
    
    #############################
    ###### Barbic version 2 #####
    #############################
    print('PCA version 2')
    print('Calculating Mahalanobis distance')
    
    #Define data size
    n_frame = angles_center_df.shape[0]
    n_axis = angles_center_df.shape[1]
    
    ## Loop over frames in data and approimate the data with the Gaussian distribution
    T = int(30/2) #number of frames expected in a small movement / 2
    K_init = int(T)
    delta = 1 #step-size between K's [frames]
    info_dt['T'] = [T]
    info_dt['delta'] = [delta]
    
    H_ls = []
    H_index = []
    
    K_index = [K for K in range(K_init, n_frame-T, delta)]
    
    #Looping over frames in data
    for i in range(n_frame):
        
        #Looping over K
        if i in K_index:
            K = i
            #print('K: ', K)
        
            #Extract data
            data = angles_center_df.iloc[:K].values
            
            #Compute SVD
            V, S = PCA_func(data)
            
            #Approximate Gaussian distribution to data - calculate covariance matrix using S and V
            C = Gaussian_func (S, V, r, n_axis, n_frame)
            
            #Calculate mahalanobis distance
            ##determine how likely the next data points is to belong to Gaus
            H = Mahalanobis_func (angles_center_df, C, K, T)
            H_ls.append(H)
            H_index.append((i,H)) #i'th frame-index and H-value (Mahalanobis distance)
            
    #Store in information dt
    info_dt['H_ls'] = [H_ls]
    info_dt['H_index'] = [H_index]
    
    ## Segmentation
    print('Segmentation')
    
    info_dt['threshold'] = [threshold]
    VP = []
    cut_index = []
    Flag = False
    
    #Looping over all Mahalanobis distances calculated above
    for i in range(1, len(H_ls) - 1):
        
        index_i = H_index[i][0] #Define the i'th frame-index
        H_val_i = H_index[i][1] #Define the Mahalanobis distance corresponding to the i'th frame-index
        
        if H_val_i != H_ls[i]:  #Error
            print('error')
            break
        
        #Peak in Mahalanobis distance values
        if H_ls[i] > H_ls[i - 1] and H_ls[i] > H_ls[i + 1]: 
            
            #If last event was a valley
            if len(VP) > 1:
                if VP[-1][0] == 'V':
                    H_diff = abs(H_ls[i] - VP[-1][1]) #Calcualte the difference in H-value
                    
                    #Cut-criteria
                    if H_diff > threshold:
                        CUT_i = i
                        cut_index.append(i)
            VP.append(('P', H_val_i, index_i))
        
        #Valley
        elif H_ls[i] < H_ls[i - 1] and H_ls[i] < H_ls[i + 1]: 
            VP.append(('V', H_val_i, index_i))
            
    print('Store')
        
    #Store in information dt
    info_dt['cut_index'] = [cut_index]
    
    #Number of cuts performed
    print('Number of CUTs: ', len(cut_index),'/', n_frame)
    info_dt['n_cuts'] = [len(cut_index)]
    info_dt['cuts_index'] = [cut_index]
    
    #Compare with new_infant index = to chech if the segmentation identify a change in infants    
    compare = [i for i in cut_index if i in new_infant_index] 
    print('Age ', str(age), 'identified ', str(len(compare)), ' infant-changes')
    
    #Store in information-directory
    info_dt['n_new_infants'] = [len(compare)]
    info_dt['New_infant_index'] = [compare]
    
    #Save segmentation index
    path_file = os.path.join(path_age, 'PCA2_index.csv')
    with open(path_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(cut_index)
    
    ## FINALLY SAVE INFORMATION DIRECTORY
    path_file = os.path.join(path_age, 'Info_dt.pkl')
    with open(path_file, 'wb') as pickle_file:
        pickle.dump(info_dt, pickle_file)
    
    print('AGE DONE')
    
    
print('FINALLY DONE')