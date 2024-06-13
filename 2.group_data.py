#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: Motion Model, Gorup data
Description: This files group data in age-groups and combine data-files to one for each age-group

Created on Wed Apr 10 17:18:27 2024
@author: anne
"""
#%% Load packages from setting
import os
path_root = '...'
os.chdir(path_root)
from settings import *
from functions import *

#%% Load file_ls for unique and age-filtered files
path_file = os.path.join(path_data_py, 'file_ls.txt')
with open(path_file, 'r') as f:
    file_ls = f.readlines()
file_ls = [int(line.strip()) for line in file_ls]

#%% Loop though data files and assign to age-groups

timestamp_df = pd.DataFrame()
age_df = pd.DataFrame()
file_index = {}
Age1_cnt = 0
Age2_cnt = 0
Age3_cnt = 0

FLAG1, FLAG2, FLAG3 = False, False, False


for file in file_ls:
    print('FILE: ' + str(file))
          
    start_time = time.time()
    
    ###############################################################################
    # LOAD DATA
    ###############################################################################
    
    ID, Age, TimeStamps, positions_df, angles_data, positions_var, angles_var = load_df_func (file)

    n_frame = len(TimeStamps)
    timestamp_df[ID] = len(TimeStamps)
    age_df[ID] = Age

    ###############################################################################
    # AGE GROUPS
    ###############################################################################
    
    if Age <= 8.6:
        file_index[file] = ['age gr = 1']
        Age1_cnt += 1
        if FLAG1 == False:
            #position1_df = positions_df
            #angles1_df = angles_data
            TimeStamp1 = TimeStamps
            n_frame1 = [len(TimeStamps)]
            Age1 = [Age]
            FLAG1 = True
        else:
            #position1_df = pd.concat([position1_df, positions_df], ignore_index=True)
            #angles1_df = pd.concat([angles1_df, angles_data], ignore_index=True)
            TimeStamp1 = np.concatenate((TimeStamp1, TimeStamps))
            n_frame1.append(len(TimeStamps))
            Age1.append(Age)
    
    elif Age > 8.6 and Age <= 17.2:
        file_index[file] = ['age gr = 2']
        Age2_cnt += 1
        if FLAG2 == False:
            #position2_df = positions_df
            #angles2_df = angles_data
            TimeStamp2 = TimeStamps
            n_frame2 = [len(TimeStamps)]
            Age2 = [Age]
            FLAG2 = True
        else:
            #position2_df = pd.concat([position2_df, positions_df], ignore_index=True)
            #angles2_df = pd.concat([angles2_df, angles_data], ignore_index=True)
            TimeStamp2 = np.concatenate((TimeStamp2, TimeStamps))
            n_frame2.append(len(TimeStamps))
            Age2.append(Age)
   
    elif Age > 17.2 and Age <= 25.8:
        file_index[file] = ['age gr = 3']
        Age3_cnt += 1
        if FLAG3 == False:
            #position3_df = positions_df
            #angles3_df = angles_data
            TimeStamp3 = TimeStamps
            n_frame3 = [len(TimeStamps)]
            Age3 = [Age]
            FLAG3 = True
        else:
            #position3_df = pd.concat([position3_df, positions_df], ignore_index=True)
            #angles3_df = pd.concat([angles3_df, angles_data], ignore_index=True)
            TimeStamp3 = np.concatenate((TimeStamp3, TimeStamps))
            n_frame3.append(len(TimeStamps))
            Age3.append(Age)
    else:
        print('Problem: ', file, ' age: ', Age)
        
    
    print('Run time: ', round((time.time()-start_time)/60,2))

#Save to file
path_agedata = os.path.join(path_data, '4.Grouped data')

position1_df.to_csv(os.path.join(path_agedata, 'position1.csv'), index=False) 
position2_df.to_csv(os.path.join(path_agedata, 'position2.csv'), index=False) 
position3_df.to_csv(os.path.join(path_agedata, 'position3.csv'), index=False) 

angles1_df.to_csv(os.path.join(path_agedata, 'angles1.csv'), index=False) 
angles2_df.to_csv(os.path.join(path_agedata, 'angles2.csv'), index=False) 
angles3_df.to_csv(os.path.join(path_agedata, 'angles3.csv'), index=False) 

#Make updated timestamps for concatenated age-group data
def TimeStamp_update_func (TimeStamp):
    TimeStamp_ls = [0]
    for i in range(1,len(TimeStamp)):
        t_old = TimeStamp[i-1]
        t_i = TimeStamp[i]
        t_new = t_old + (t_i - t_old)
        TimeStamp_ls.append(t_new)
    return(TimeStamp_ls)

TimeStamp11 = TimeStamp_update_func(TimeStamp1) 
TimeStamp21 = TimeStamp_update_func(TimeStamp2)
TimeStamp31 = TimeStamp_update_func(TimeStamp3)

np.savetxt(os.path.join(path_agedata, 'TimeStamp1.csv'), TimeStamp11, delimiter=',', fmt='%d')
np.savetxt(os.path.join(path_agedata, 'TimeStamp2.csv'), TimeStamp21, delimiter=',', fmt='%d')
np.savetxt(os.path.join(path_agedata, 'TimeStamp3.csv'), TimeStamp31, delimiter=',', fmt='%d')

Age1 = np.array(Age1)
Age2 = np.array(Age2)
Age3 = np.array(Age3)
np.savetxt(os.path.join(path_agedata, 'Age1.csv'), Age1, delimiter=',')
np.savetxt(os.path.join(path_agedata, 'Age2.csv'), Age2, delimiter=',')
np.savetxt(os.path.join(path_agedata, 'Age3.csv'), Age3, delimiter=',')

#New infant index
def new_infant_index_func(n_frame):
    new_infant_index = [0]
    for i in range(len(n_frame) - 1):
        n_frame_i = n_frame[i]
        index_prev = new_infant_index[i]
        index = index_prev + n_frame_i
        new_infant_index.append(index)
    return(new_infant_index)
    
new_infant_index1 = new_infant_index_func(n_frame1)
new_infant_index2 = new_infant_index_func(n_frame2)
new_infant_index3 = new_infant_index_func(n_frame3)

np.savetxt(os.path.join(path_agedata, 'new_infant_index1.csv'), new_infant_index1, delimiter=',', fmt='%d')
np.savetxt(os.path.join(path_agedata, 'new_infant_index2.csv'), new_infant_index2, delimiter=',', fmt='%d')
np.savetxt(os.path.join(path_agedata, 'new_infant_index3.csv'), new_infant_index3, delimiter=',', fmt='%d')