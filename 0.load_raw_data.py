#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: Load raw data

Description: 
This file loads raw mat-files from 'Data/Data_raw'
The data is re-organised and stored as directories
The data-directories are stores as pkl-files in 'Data/Data_dup_py'
The data-files containing unique data is stores as pkl-files in 'Data/Data_py'

Created on Sun Jan 28 15:15:34 2024
@author: anne
"""

#%% Load packages from setting
import os
path_root = '...'
os.chdir(path_root)
from settings import *
from functions import *

#%% Initialise
num_pat = 300 #number of samples/patients/infants
info_df = pd.DataFrame(index = range(num_pat), columns = ['Filename','ID','Age','Frames'])

#%%# Loop through the files
for i in range(1, num_pat+1):
    
    print(i)
    
    # Construct the file name
    filename = f'{i:03d}'
    path_file = os.path.join(path_data, '1.Data_raw/' +  filename + '.mat')

    try:
        # Load the mat-file
        mat_data = scipy.io.loadmat(path_file)
        mat_human = mat_data.get('Human')

        #### Extract data ####
        
        ## Patient ID and age
        ID = mat_human['ID'][0,0][0,0]
        Age = mat_human['Age'][0,0][0,0]

        ## Timestamp vector
        TimeStamp_vec = mat_human['TimeStamps'][0,0] 
        
        ## Angles matrix
        Angles_mat = mat_human['Angles'][0,0]
        
        ## Pose data
        Poses = mat_human['Poses'][0,0]
        #Poses.dtype.names #= ('Positions', 'Variables', 'Bases')
        pose_pos_dt  = load_raw_dict_func(Poses['Positions'][0,0])
        pose_var_dt  = load_raw_dict_func(Poses['Variables'][0,0])
        pose_base_dt = load_raw_dict_func(Poses['Bases'][0,0])
        
        ## Save filename, ID and age in dataframe 
        info_df.loc[i] = [filename, ID, Age, len(TimeStamp_vec[0])]

        #### Combine data for the patient into a directory ####
        data = {'ID'       : ID, 
                'Age'      : Age,
                'TimeStamp': TimeStamp_vec,
                'Angles'   : Angles_mat,
                'Poses'    : {'Position'  : pose_pos_dt,
                              'Variables' : pose_var_dt,
                              'Bases'     : pose_base_dt}} 
        
        
        ## Save data dictionary
        path_file = os.path.join(path_data, '2.Data_dup_py/' + filename + '.pkl')
        with open(path_file, 'wb') as pickle_file:
            pickle.dump(data, pickle_file)
               
        ## Store data dicretory as key-value relationship in large directory
        #data_dt[filename] = data
        
    except Exception as e:
        print(f"Error loading {filename}: {e}")

#%% Save files

## Save info dataframe to excel-file
info_df = info_df.dropna(how='all')
path_file = os.path.join(path_data, '2.Data_dup_py/' + 'information.xlsx')
info_df.to_excel(path_file, index = True, header = True)
#del info_df

#%% DUPLICATED DATA
#%% Find file-names with duplicated data

info_unique_df = info_df.drop_duplicates(subset=['ID', 'Age'])
file_unique_ls = info_unique_df['Filename'].tolist()

#%% Move the unique files to "Data_py" folder

import shutil

for filename in file_unique_ls:
    print(filename)
    
    #Define paths
    path_from = os.path.join(path_data, '2.Data_dup_py/' + filename + '.pkl')
    path_to = os.path.join(path_data, '3.Data_py/')
    
    #Copy files
    shutil.copy(path_from, path_to)

# Save the unique info dataframe to excel-file
path_data_py = os.path.join(path_data, '3.Data_py/')
path_file = os.path.join(path_data_py, 'information_unique.xlsx')
info_unique_df.to_excel(path_file, index = True, header = True)
#del info_unique_df
    
# Save the filenames of the unique files to txt-file
path_file = os.path.join(path_data_py, 'file_unique_ls.txt')
with open(path_file, 'w') as file:
    for item in file_unique_ls:
        file.write("%s\n" % item)
        
#%% AGE FILTER

# Remove patients with age > 6 months from information and file_ls
age_df = info_unique_df[info_unique_df['Age'] <= 52/2]

# Remove patients with age < 5 weeks (due to lack of data) from information and file_ls
info_age_df = age_df[age_df['Age'] > 5]

file_age_ls = list(info_age_df['Filename'])

# Save the unique and age-corrected info dataframe to excel-file
path_file = os.path.join(path_data_py, 'information.xlsx')
info_age_df.to_excel(path_file, index = True, header = True)
    
# Save the filenames of the unique and age-corrected files to txt-file
path_file = os.path.join(path_data_py, 'file_ls.txt')
with open(path_file, 'w') as file:
    for item in file_age_ls:
        file.write("%s\n" % item)


