#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Title: Age Prediction Model, Feature extraction
Description: 
This file calculates features and seperate data into a hold-out data set and a training data set

Created on Sun Jan 28 16:55:43 2024
@author: anne
"""

#%% Load packages from setting
import os
path_root = '...'
os.chdir(path_root)
from settings import *
from functions import *

#%%##################
# EXTRACT FEATURES ##
#####################

"""The first part of the file calculates features for one file at a time
The features are saved as dicts in folder Feature_py"""

#%% Initialise

frame_width = 1      #[timesteps]
overlap = 5          #[frames] 

data_dt = {}

#Load file_ls for unique and age-filtered files
path_file = os.path.join(path_data_py, 'file_ls.txt')
with open(path_file, 'r') as f:
    file_ls = f.readlines()
file_ls = [int(line.strip()) for line in file_ls]

#%% Feature extraction lopp

for file in file_ls:
    print('FILE: ' + str(file))
          
    start_time = time.time()
    
    ###############################################################################
    # LOAD DATA
    ###############################################################################
    
    file_dt = {}
    
    ID, Age, TimeStamps, positions_df, angles_df, positions_var, angles_var = load_df_func (file)

    #Define dimensions
    n_timestep = len(TimeStamps)                         #5607 (depend on the recording
    n_frame  = int(n_timestep/frame_width)               #Number of frames
    n_pos_var = len(positions_var)                       #19
    n_ang_var = len(angles_var)                          #13    
    
    #Store
    file_dt['ID']            = ID
    file_dt['Age']           = Age
    file_dt['TimeStamps']    = TimeStamps
    file_dt['positions_raw'] = positions_df
    file_dt['angles_raw']    = angles_df
    
    #############################
    # NORMALISING POSITION DATA #
    #############################
    
    positions_raw = positions_df.copy()
    RLA_len_vec = np.empty(n_frame)

    #Calculate length of right lower arm 
    for i in range(n_frame):
        RLA_col = positions_raw.filter(regex = '^' + 'RLArm')
        RH_col = positions_raw.filter(regex = '^' + 'RHand')
        RLA_len_vec[i] = abs(np.linalg.norm(np.array(RH_col.iloc[i]) - np.array(RLA_col.iloc[i])))
    RLA_len = np.mean(RLA_len_vec)
    
    #Normalise position data
    ## Subtract BodyCenter and divide with length of right lower arm
    for i in range(len(positions_raw.columns)):
        var = positions_raw.columns[i]
        if '1' in var: 
            positions_df[var] = (positions_raw[var] - positions_raw['BodyCenter1']) / RLA_len
        elif '2' in var: 
            positions_df[var] = (positions_raw[var] - positions_raw['BodyCenter2']) / RLA_len
        elif '3' in var: 
            positions_df[var] = (positions_raw[var] - positions_raw['BodyCenter3']) / RLA_len
    
    #Store
    file_dt['Norm'] = RLA_len
    file_dt['positions_norm'] = positions_df
    
    ###############################################################################
    # WINDOW-LOOP
    ###############################################################################
    
    #Loop over window-sizes
    for window_width in [30, 60]:     
        #[frames] = 3 frames [10 timesteps] * 10 timesteps [30/ms] = 1 sec
        #[frames] = 6 frames [10 timesteps] * 10 timesteps [30/ms] = 2 sec
        
        print('WINDOW SIZE: ', str(window_width*frame_width))  

        #Define dimensions   
        n_window = int(n_frame/(window_width-overlap))       #Number of windows
        init_dt = {'n_timestep' : n_timestep, 'n_frame': n_frame, 'n_window': n_window, 'frame_width' : frame_width, 'window_width': window_width, 'overlap': overlap, 'n_pos_var': n_pos_var, 'n_ang_var': n_ang_var}
        
        #Store
        file_dt['window_size'] = window_width*frame_width
        
        ###############################################################################
        # POSITION-FEATURES
        ###############################################################################
        print('Position features')
        
        #Displacement, velocity, acceleration, and jerkiness    
        disp_f_df, vel_f_df, acc_f_df, jerk_f_df = feature_VelAcc_func (positions_df, TimeStamps, positions_var, init_dt)    
       
        # Range, mean, variation, and entropy of position-features
    
        #Calculate statistics within each window
        disp_w_dt = feature_stat_func (disp_f_df, positions_var, init_dt)
        vel_w_dt = feature_stat_func (vel_f_df, positions_var, init_dt)
        acc_w_dt = feature_stat_func (acc_f_df, positions_var, init_dt)
        jerk_w_dt = feature_stat_func (jerk_f_df, positions_var, init_dt)
        
        #Store
        file_dt['pos_disp_' + str(window_width) + '_dt'] = disp_w_dt
        file_dt['pos_vel_' + str(window_width) + '_dt']  = vel_w_dt
        file_dt['pos_acc_' + str(window_width) + '_dt']  = acc_w_dt
        file_dt['pos_jerk_' + str(window_width) + '_dt'] = jerk_w_dt
           
        ###############################################################################
        # ANGLES-FEATURES
        ###############################################################################
        print('Angle features')
        
        # Rotation
        ang_vel_f_df, ang_acc_f_df = feature_rotation_func (angles_df, TimeStamps, angles_var, init_dt)
    
        # Calculate range, variation, and entropy of angles-features (within each window)
        vel_w_dt = feature_stat_func (ang_vel_f_df, angles_var, init_dt)
        
        #Store 
        file_dt['ang_vel_' + str(window_width)+ '_dt'] = vel_w_dt
        #file_dt['ang_acc_' + str(window_width)]   = acc_w_df
    
        ###############################################################################
        # SPECIAL FEATURES
        ###############################################################################
        print('Special features')
    
        ############
        # DISTANCE #
        ############
        
        dist_var = ['Body', 'Head',
                    'RUArm', 'LUArm', 'RLArm', 'LLArm', 'RHand', 'LHand', 
                    'RULeg', 'LULeg', 'RLLeg', 'LLLeg', 'RFoot', 'LFoot']
        
        #Initialise
        column_min   = [i + '_min_' + str(window_width) for i in ['dist_sym', 'dist_head']]
        column_max   = [i + '_max_' + str(window_width) for i in ['dist_sym', 'dist_head']]
        column_mean  = [i + '_mean_' + str(window_width) for i in ['dist_sym', 'dist_head']]
        column_var   = [i + '_var_' + str(window_width) for i in ['dist_sym', 'dist_head']]
        column_entro = [i + '_entro_' + str(window_width) for i in ['dist_sym', 'dist_head']]
        
        min_df   = pd.DataFrame(columns = column_min, index = dist_var)
        max_df   = pd.DataFrame(columns = column_max, index = dist_var)
        mean_df  = pd.DataFrame(columns = column_mean, index = dist_var)
        var_df   = pd.DataFrame(columns = column_var, index = dist_var)
        entro_df = pd.DataFrame(columns = column_entro, index = dist_var)
        
        
        # Calculate symmetrical distance
        sym_pair = [['BodyCenter', 'BodyCenter'],['HeadCenter', 'HeadCenter'], 
                    ['LUArm', 'RUArm'], ['LUArm', 'RUArm'], ['LLArm', 'RLArm'], ['LLArm', 'RLArm'], ['LHand', 'RHand'], ['LHand', 'RHand'],
                    ['LULeg', 'RULeg'], ['LULeg', 'RULeg'], ['LLLeg', 'RLLeg'], ['LLLeg', 'RLLeg'], ['LFoot', 'RFoot'], ['LFoot', 'RFoot']]
       
        dist_sym_f_df = feature_dist_func (positions_df, dist_var, sym_pair, init_dt) 
        
        
        # Calculate distance to face
        head_pair = [['HeadCenter', 'BodyCenter'], ['HeadCenter', 'HeadCenter'],
                     ['HeadCenter', 'RUArm'], ['HeadCenter', 'LUArm'] ,['HeadCenter', 'RLArm'], ['HeadCenter', 'LLArm'], ['HeadCenter', 'RHand'], ['HeadCenter', 'LHand'], 
                     ['HeadCenter', 'RULeg'], ['HeadCenter', 'LULeg'], ['HeadCenter', 'RLLeg'], ['HeadCenter', 'LLLeg'], ['HeadCenter', 'RFoot'], ['HeadCenter', 'LFoot']]
        
        
        dist_head_f_df = feature_dist_func (positions_df, dist_var, head_pair, init_dt)
            
        # Calculate range, variation, and entropy of distance-measures (within each window)
        dist_sym_w_df = feature_stat_func (dist_sym_f_df, dist_var, init_dt)
        dist_head_w_df = feature_stat_func (dist_head_f_df, dist_var, init_dt)
    
        #Store
        file_dt['dist_sym_' + str(window_width) + '_dt']   = dist_sym_w_df
        file_dt['dist_head_' + str(window_width) + '_dt']  = dist_head_w_df
        
        #Calculate head-extremity frequency
       
        #Calculate head-size
        neck_col = positions_df.filter(regex = '^' + 'HeadJoint')
        head_col = positions_df.filter(regex = '^' + 'HeadCenter')
        head_r = abs(np.linalg.norm(np.array(neck_col.iloc[100]) - np.array(head_col.iloc[100])))
        #Define a near-head measure
        near_head = head_r + (head_r/4) 
        
        #Count the number of times an extremity is near the face
        cross_df = pd.DataFrame(columns = dist_var)
        for var in dist_var:
            data = dist_head_f_df[var]
            Flag = False
            cross = 0
            for i in range(len(data)):
                #If extremity moves near the head and hasn't been there lately
                if (data[i] <= near_head) and Flag == False:
                    cross += 1
                    Flag = True
                #If extremity moves away from the head and has been near the head lately
                elif (data[i] > near_head) and Flag == True:
                    Flag = False
            cross_df[var] = [cross]
        cross_df = cross_df.transpose()
        
        #Calculate the number of windows an extremity is near the face (time-estimate) 
        windof_df = (dist_head_f_df <= near_head).sum()
        
        #Combine and adjust for the number of windows (length of recording)
        dist_head_cnt  = pd.concat([cross_df/n_window, windof_df/n_window], axis=1, keys = ['Cross_adj', 'Window_adj'])
        
        #Store
        file_dt['dist_head_cnt_' + str(window_width) + '_df'] = np.transpose(dist_head_cnt)
    
        ##########
        # VOLUME #
        ##########

        #Calculate volume of movement
        # ( normalise position-data by subtracting the parent joint from the child joint )
        vol_df = feature_vol_func (positions_df, positions_var, init_dt)
        
        #Store
        file_dt['volume_' + str(window_width) + '_df'] = vol_df 
        
        ###########
        # CLUSTER #
        ###########
    
        #Data-representation: mean-vector of window-sized subsets
        subset = np.array_split(angles_df.values, n_window)[:-3]
        
        angles_sub = []
        for data in subset:
            angles_sub.append(np.mean(data, axis = 0))    
        angles_sub = np.array(angles_sub)
        
        #Iterate over number of clusters
        
        max_cluster = angles_sub[:,0].shape[0]
        ss_dist_ls = []         #Sum of squared Euclidean distance - Within cluster variance 
        sil_score_ls = []       #Silhouette score - used to evalute performance
        best_sil = -1
        
        for n_cluster in range(2, max_cluster):
                
            #K-means clustering
            kmeans = KMeans(n_cluster)
            kmeans.fit(angles_sub)
            labels = kmeans.labels_
            ss_dist_ls.append(kmeans.inertia_)
            sil_score = silhouette_score(angles_sub, labels)
            sil_score_ls.append(sil_score)
            
            #Evaluate
            if sil_score > best_sil:
                best_sil  = sil_score
                best_dist = kmeans.inertia_
                best_clusters = n_cluster
        
        cluster_df = pd.DataFrame(data = {'n_cluster': [best_clusters], 'silh': [best_sil], 'distance': [best_dist]})
        
        #Store:
        file_dt['cluster_' + str(window_width) + '_df'] = cluster_df
        
    ###############################################################################
    # SAVE
    ###############################################################################
    print('Save')

    ## Save data dictionary
    filename = f'{file:03d}'
    #filename = file
    path_file = os.path.join(path_feature, filename + '.pkl')
    with open(path_file, 'wb') as pickle_file:
        pickle.dump(file_dt, pickle_file)
    
    #Calculate runtime
    print('Run time: ', round((time.time()-start_time)/60,2))


    
#%%###############
# HOLD-OUT DATA ##
##################
"""The second part of the file seperate the data into a hold-out data set and a training data set"""

#%% Load information about data

#Load information as data frame
path_file = os.path.join(path_root, 'information.xlsx')
info_df = pd.read_excel(path_file)

#%% Hold out data
#The hold out data consist of 10% of the main data - uniform distributed over age

num_hold = np.floor(len(info_df['ID'])*0.10) #10% = 15 patients

#Sort information according to age
age_sorted = info_df.sort_values(by='Age')

#Extract hold out data = every 10th ID 
info_out_df = age_sorted[9::10]  
file_out_ls = info_out_df['Filename'].tolist()

histplot_func (info_out_df['Age'], 'Age distribution of hold out', labels = ['Age [weeks]',''], nbins = 'auto')

#File list without holdout ID
info_in_df = info_df[~info_df['ID'].isin(info_out_df['ID'].tolist())]
file_in_ls = info_in_df['Filename'].tolist()

#%% Move "hold-in" data files to new directory

for file in file_in_ls:
    filename = f'{file:03d}'
    print(filename)
    
    #Define paths
    path_from = os.path.join(path_data, '3.Data_py/' + filename + '.pkl')
    path_to = os.path.join(path_data, '4.Data_ML_py/')
    
    #Copy files
    shutil.copy(path_from, path_to)

# Save the info dataframe of "hold-in" files to excel
path_data_py = os.path.join(path_data, '4.Data_ML_py/')
path_file = os.path.join(path_data_py, 'information.xlsx')
info_in_df.to_excel(path_file, index = True, header = True)
    
# Save the filenames of "hold-in" files to txt-file
path_file = os.path.join(path_data_py, 'file_ls.txt')
with open(path_file, 'w') as file:
    for item in file_in_ls:
        file.write("%s\n" % item)
        
        
#%% Move "hold-out" data files to new directory

for file in file_out_ls:
    filename = f'{file:03d}'
    print(filename)
    
    #Define paths
    path_from = os.path.join(path_data, '3.Data_py/' + filename + '.pkl')
    path_to = os.path.join(path_data, '4.Data_ML_py/', '0.Hold_out_py')
    
    #Copy files
    shutil.copy(path_from, path_to)

# Save the info dataframe of "hold-out" files to excel
path_data_py = os.path.join(path_data, '4.Data_ML_py/', '0.Hold_out_py')
path_file = os.path.join(path_data_py, 'information.xlsx')
info_out_df.to_excel(path_file, index = True, header = True)
    
# Save the filenames of "hold-out" files to txt-file
path_file = os.path.join(path_data_py, 'file_ls.txt')
with open(path_file, 'w') as file:
    for item in file_out_ls:
        file.write("%s\n" % item)
 