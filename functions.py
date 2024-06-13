#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: Functions
Description: This file define functions

Created on Sun Jan 28 16:55:43 2024
@author: anne
"""

#%% Load packages from setting
import os
path_root = '...'
os.chdir(path_root)
from settings import *

#%%############################
# FUNCTIONS FOR LOADING FILES #
###############################

#%% Load raw data into directory function

#Example
#pose_pos_dt  = load_raw_dict_func(Poses['Positions'][0,0])

def load_raw_dict_func(data):
    
    """ This function loads raw data into a diretory"""
    
    data_dt = {}
    
    for data_type in data.dtype.names:
        data_dt[data_type] = data[data_type]
    
    return(data_dt)


#%% Extract data from directory

#Example
#ID, Age, TimeStamps, positions_df, angles_df, positions_var, angles_var = load_df_func (file)

def load_df_func (file):
    #Previous: extract_data_func
    
    """This function extracts ID, Age, Timestamps, Angles and Positions 
    from the directory (created in load_raw_data.py) belogning to 'file' 
    Angles and Positions are saved as dataframes"""
    
    #Load file_ls
    path_file = os.path.join(path_data_py, 'file_ls.txt')
    with open(path_file, 'r') as f:
        file_ls = f.readlines()
    file_ls = [int(line.strip()) for line in file_ls]
    
    #Define column names for angle-data
    angles_col_names = ['BodyAngle1',  'BodyAngle2',  'BodyAngle3',
                        'HeadAngle1',  'HeadAngle2', 'HeadAngle3', 
                        'LUArmAngle1', 'LUArmAngle2', 'LUArmAngle3', 
                        'RUArmAngle1', 'RUArmAngle2', 'RUArmAngle3', 
                        'LLArmAngle1', 'LLArmAngle2', 'LLArmAngle3', 
                        'RLArmAngle1', 'RLArmAngle2', 'RLArmAngle3', 
                        'LULegAngle1', 'LULegAngle2', 'LULegAngle3', 
                        'RULegAngle1', 'RULegAngle2', 'RULegAngle3', 
                        'LLLegAngle1', 'LLLegAngle2', 'LLLegAngle3', 
                        'RLLegAngle1', 'RLLegAngle2', 'RLLegAngle3', 
                        'LFootAngle1', 'LFootAngle2', 'LFootAngle3', 
                        'RFootAngle1', 'RFootAngle2', 'RFootAngle3']
    
    position_col_name = ['HeadJoint1',  'HeadJoint2',  'HeadJoint3', 
                         'HeadCenter1', 'HeadCenter2', 'HeadCenter3', 
                         'RUArmStart1', 'RUArmStart2', 'RUArmStart3', 
                         'RLArmStart1', 'RLArmStart2', 'RLArmStart3', 
                         'RHandStart1', 'RHandStart2', 'RHandStart3', 
                         'LUArmStart1', 'LUArmStart2', 'LUArmStart3', 
                         'LLArmStart1', 'LLArmStart2', 'LLArmStart3', 
                         'LHandStart1', 'LHandStart2', 'LHandStart3', 
                         'RULegStart1', 'RULegStart2', 'RULegStart3', 
                         'RLLegStart1', 'RLLegStart2', 'RLLegStart3', 
                         'RFootStart1', 'RFootStart2', 'RFootStart3', 
                         'RToesStart1', 'RToesStart2', 'RToesStart3', 
                         'LULegStart1', 'LULegStart2', 'LULegStart3', 
                         'LLLegStart1', 'LLLegStart2', 'LLLegStart3', 
                         'LFootStart1', 'LFootStart2', 'LFootStart3', 
                         'LToesStart1', 'LToesStart2', 'LToesStart3', 
                         'Chest1',      'Chest2',      'Chest3', 
                         'Crotch1',     'Crotch2',     'Crotch3', 
                         'BodyCenter1', 'BodyCenter2', 'BodyCenter3']
    
    #Load directory
    #file = file_ls[file]
    #filename = f'{file:03d}'
    filename = file
    path_file = os.path.join(path_data_py, filename + '.pkl')
    with open(path_file, 'rb') as file:
        data_dt = pickle.load(file)
    
    #Extract info-data
    ID = data_dt['ID']
    Age = data_dt['Age']
    TimeStamp = data_dt['TimeStamp'][0]
    
    #Extract angles data
    angles_mat = data_dt['Angles']    
    angles_df = pd.DataFrame(np.transpose(angles_mat), columns=angles_col_names)
    
    #Extrast position data
    position_dt = data_dt['Poses']['Position']
    position_keys = [key for key in position_dt.keys()]
    position_mat = np.zeros((len(TimeStamp), len(position_keys)*3))
    
    for i in range(0,len(position_keys)):
        key = position_keys[i]
        #print(key)
        position_var = position_dt[key]
        for j in range(len(position_var)):
            position_mat[j,i*3:(i*3)+3] = np.transpose(position_var[j][0])
        
    positions_df = pd.DataFrame(position_mat, columns=position_col_name)
    
    #Define a list of all position and angles columnnames
    positions_var = list(set([i[:-1] for i in positions_df.columns.tolist()]))
    positions_var = [var.replace("Start", "") for var in positions_var] #remove "start" in end of some variable-names
    positions_var = sorted(positions_var)
    angles_var = list(set([i[:-1] for i in angles_df.columns.tolist()]))
    angles_var = sorted(angles_var)
    
    return(ID, Age, TimeStamp, positions_df, angles_df, positions_var, angles_var)

#%% Rotation function (for nomalising data)
def rotation_func (positions_df):
    
    """ This function rotates the position data"""
    
    positions_raw = positions_df.copy()
    
    ##Calculate angle of rotation around z-axis based on chest-joint
    
    #Extract BodyCenter and Chest position
    BodyCenter = positions_raw.filter(regex = '^' + 'BodyCenter')
    bodycenter_x = positions_df['BodyCenter1'].values
    bodycenter_y = positions_df['BodyCenter2'].values
    bodycenter_z = positions_df['BodyCenter3'].values
    
    Chest = positions_raw.filter(regex = '^' + 'Chest')
    chest_x = positions_df['Chest1'].values
    chest_y = positions_df['Chest2'].values
    chest_z = positions_df['Chest3'].values
    
    n_frame = len(chest_x)
    
    #Initialise
    positions_rot_df = pd.DataFrame(columns = positions_df.columns)
    pos_col = [col[:-1] for col in positions_df.columns]
    pos_col = list(dict.fromkeys(pos_col))
    
    #LOOP
    for var in pos_col: 
        #print(var)
        #Extract data
        selected_col = positions_df.filter(regex = '^' + var).values
        
        var1 = np.zeros(n_frame)
        var2 = np.zeros(n_frame)
        var3 = np.zeros(n_frame)
        
        for i in range(n_frame):
            selected_row = selected_col[i]
            
            ##Calculate theta

            #Calculate length of vector between bodycenter and chest
            chest_vec = math.sqrt((chest_x[i] - bodycenter_x[i])**2 + (chest_y[i] - bodycenter_y[i])**2)
            
            #Define position of rotated chest
            chest_x_rot = -abs(chest_vec) #position on -x-axis = the negative absolute length of the vector between bodycenter and chest
            chest_y_rot = 0 #align with x-axis
            chest_z_rot = chest_z[i] #Original z-values
            
            #Calculate angle of rotation (theta) in radians
            v = [chest_x[i], chest_y[i]]
            w = [chest_x_rot, chest_y_rot]
            
            #theta = math.atan2(chest_x_rot - chest_x[i], chest_y_rot - chest_y[i])
            #theta1 = math.atan2(chest_x[i] - chest_x_rot, chest_y[i] - chest_y_rot)
            theta = math.atan2(w[1]*v[0] - w[0]*v[1],  w[0]*v[0] + w[1]*v[1])
            #math.degrees(theta2)
            
            ## Rotation
            
            #Define rotation matrix
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                        [np.sin(theta), np.cos(theta)]])
            
            #Rotate data
            rotated_point = np.dot(rotation_matrix, selected_row[:2]) #select x and y element from i-th frame from the selected variable
                    
            var1[i] = rotated_point[0]
            var2[i] = rotated_point[1]
            var3[i] = selected_row[-1] #Save the original z-value
            
        positions_rot_df[var + '1'] = var1
        positions_rot_df[var + '2'] = var2
        positions_rot_df[var + '3'] = var3
    
    return(positions_rot_df)

#%%#################################
# FUNCTIONS FOR FEATURE EXTRACTION #
####################################

#%% Velocity-acceleration function

#Example
#disp_f_df, vel_f_df, acc_f_df, jerk_f_df = feature_VelAcc_func (positions_df, TimeStamps, positions_var, init_dt) 

def feature_VelAcc_func (data_df, TimeStamps, var_ls, init_dt):
    
    """ This function calculates the disposition, velocity, acceleration, and jerkiness from position data 
    The measures are calculated between frames in data.
    The measures are saved in dataframes"""

    #Initiate
    n_timestep = init_dt['n_timestep']                #Number of timesteps in the recording
    n_frame  = init_dt['n_frame']                     #Number of frames in the data [timesteps]
    n_window = init_dt['n_window']                    #Number of windows in the data [frames]
    frame    = init_dt['frame_width']
    window   = init_dt['window_width']                #Window width [frames]
    overlap  = init_dt['overlap']                     #Overlap width [frames]
    
    disp_f_df = pd.DataFrame(columns = var_ls)
    vel_f_df  = pd.DataFrame(columns = var_ls)
    acc_f_df  = pd.DataFrame(columns = var_ls)
    jerk_f_df = pd.DataFrame(columns = var_ls)
    
    for var in var_ls:
        
        #Extract data
        data = data_df.filter(regex = '^' + var)
        
        # FRAME
        
        #Initialise
        delta_t             = np.empty(n_frame)
        displacement_frame  = np.full(n_frame, np.nan)
        velocity_frame      = np.full(n_frame, np.nan)
        acceleration_frame  = np.full(n_frame, np.nan)
        jerk_frame          = np.full(n_frame, np.nan)

        #Make an index of frame-positions in time
        frame_index = [i for i in range(frame, n_timestep, frame)]
        
        #Calculate the movement-features for each frame = x timestamps 
        for i in range(len(frame_index)):
            frame_i = frame_index[i]      #the position in data of the i-th window
            
            #Calculate the distance travelled on the three axis between the i-1 and the i'th frame
            delta_1 = abs((data.iloc[frame_i, 0] - data.iloc[frame_i - frame, 0]))
            delta_2 = abs((data.iloc[frame_i, 1] - data.iloc[frame_i - frame, 1]))
            delta_3 = abs((data.iloc[frame_i, 2] - data.iloc[frame_i - frame, 2]))
            
            #Calculate the time-difference between the i-1 and i'th frame
            delta_t[i] = abs(TimeStamps[frame_i] - TimeStamps[frame_i - frame])                                                #Contain nan as the first element
            
            #Calculate the features between the i-1 and the i'th frame
            ##Calculate the displacement 
            displacement_frame[i] = np.linalg.norm([delta_1, delta_2, delta_3])
            
            ##Calculate the velocity
            velocity_frame[i] = abs(displacement_frame[i] / delta_t[i])
            #velocity_frame[i] = np.linalg.norm([delta_1/delta_t[i], delta_2/delta_t[i], delta_3/delta_t[i]]) #Contain nan as the first element
            
            ##Calculate the acceleration 
            acceleration_frame[i] = abs((velocity_frame[i] - velocity_frame[i - 1]) / (delta_t[i]))          #Contain nan as the first two elements
            
            ##Calculate the 'jerkiness' 
            jerk_frame[i] = abs((acceleration_frame[i] - acceleration_frame[i - 1]) / (delta_t[i]))          #Contain nan as the first three elements
        
        #Store 
        disp_f_df[var] = displacement_frame
        vel_f_df[var]  = velocity_frame
        acc_f_df[var]  = acceleration_frame
        jerk_f_df[var] = jerk_frame
        
           
    return(disp_f_df, vel_f_df, acc_f_df, jerk_f_df)

#%% Angular-velocity function

#Example
#ang_vel_f_df, ang_acc_f_df = feature_rotation_func (angles_df, TimeStamps, angles_var, init_dt)

def feature_rotation_func (data_df, TimeStamps, var_ls, init_dt):
    
    """ This function calculates the rotation as the angular velocity
    The measures are calculated between frames in data.
    The measures are saved in dataframes"""
    
    #Initiate
    n_timestep = init_dt['n_timestep']                #Number of timesteps in the recording
    n_frame  = init_dt['n_frame']                     #Number of frames in the data [timesteps]
    #n_window = init_dt['n_window']                    #Number of windows in the data [frames]
    frame    = init_dt['frame_width']
    #window   = init_dt['window_width']                #Window width [frames]
    #overlap  = init_dt['overlap']                     #Overlap width [frames]
    
    vel_f_df  = pd.DataFrame(columns = var_ls)
    acc_f_df  = pd.DataFrame(columns = var_ls)
    
    for var in var_ls:  
        
        #Extract data
        data = data_df.filter(regex = '^' + var)
        
        # FRAME
        
        #Initialise
        delta_t            = np.full(n_frame, np.nan)
        velocity_frame     = np.full(n_frame, np.nan)
        acceleration_frame = np.full(n_frame, np.nan)

        #Make an index of frame-positions in time
        frame_index = [i for i in range(frame, n_timestep, frame)]
        
        #Calculate the movement-features for each frame = x timestamps 
        for i in range(len(frame_index)):
            frame_i = frame_index[i]      #the position in data of the i-th window
            
            #Calculate the distance travelled on the three axis between the i-1 and the i'th frame
            delta_1 = abs((data.iloc[frame_i, 0] - data.iloc[frame_i - frame, 0]))
            delta_2 = abs((data.iloc[frame_i, 1] - data.iloc[frame_i - frame, 1]))
            delta_3 = abs((data.iloc[frame_i, 2] - data.iloc[frame_i - frame, 2]))
            
            #Calculate the time-difference between the i-1 and i'th frame
            delta_t[i] = abs(TimeStamps[frame_i] - TimeStamps[frame_i - frame])                                                #Contain nan as the first element
            
            #Calculate the features between the i-1 and the i'th frame
            ##Calculate the angular velocity 
            velocity_frame[i] = np.linalg.norm([delta_1/delta_t[i], delta_2/delta_t[i], delta_3/delta_t[i]])
            
            ##Calculate the acceleration 
            acceleration_frame[i] = abs((velocity_frame[i] - velocity_frame[i - 1]) / (delta_t[i]))          #Contain nan as the first two elements
            
        #Store
        vel_f_df[var]  = velocity_frame
        acc_f_df[var]  = acceleration_frame
        
    return(vel_f_df, acc_f_df)

#%% Distance between bodyparts function

#Example
#dist_sym_f_df = feature_dist_func (positions_df, dist_var, sym_pair, init_dt) 

def feature_dist_func (data_df, var_ls, pair_ls, init_dt):
    
    """ This function calculates the distance beween body parts.
    The measures are calculated between frames in data.
    The measures are saved in dataframes"""
    
    #Initialise
    n_timestep = init_dt['n_timestep']                #Number of timesteps in the recording
    n_frame  = init_dt['n_frame']                     #Number of frames in the data [timesteps]
    n_window = init_dt['n_window']                    #Number of windows in the data [frames]
    frame    = init_dt['frame_width']
    window   = init_dt['window_width']                #Window width [frames]
    overlap  = init_dt['overlap']                     #Overlap width [frames]
    
    dist_f_df = pd.DataFrame(columns = var_ls)
    dist_w_df = pd.DataFrame(columns = var_ls)
        
    for i in range(len(var_ls)):
        #Define variables
        var = var_ls[i]
        var_pair = pair_ls[i]
        
        #Extract data
        var1 = var_pair[0]
        var2 = var_pair[1]
        data1 = data_df.filter(regex = '^' + var1)
        data2 = data_df.filter(regex = '^' + var2)
        
        #Initialise
        dist_f_vec = np.empty(n_frame)
    
        #Make an index of frame-positions in time
        frame_index = [j for j in range(1, n_timestep, frame)]
        
        #Calculate the distance between var1 and var2 on the three axis in the i'th frame
        for j in range(n_frame-1):
            frame_j = frame_index[j]      #the position in data of the i-th window
            
            var1 = np.array([data1.iloc[frame_j,0],data1.iloc[frame_j,1],data1.iloc[frame_j,2]])
            var2 = np.array([data2.iloc[frame_j,0],data2.iloc[frame_j,1],data2.iloc[frame_j,2]])
            dist_f_vec[j] = np.linalg.norm(var2 - var1)
           
        #Store
        dist_f_df[var] = dist_f_vec
    
    return(dist_f_df)

#%% Volume of movement function

#Example
#vol_df = feature_vol_func (positions_df, positions_var, init_dt)

def feature_vol_func (data_df, var_ls, init_dt):
    
    """ This function calculated the volume the movements of a joint
    The result is saved in a data frame """
    
    #Initialise
    n_timestep = init_dt['n_timestep']                #Number of timesteps in the recording
    n_frame  = init_dt['n_frame']                     #Number of frames in the data [timesteps]
    n_window = init_dt['n_window']                    #Number of windows in the data [frames]
    frame    = init_dt['frame_width']
    window   = init_dt['window_width']                #Window width [frames]
    overlap  = init_dt['overlap']                     #Overlap width [frames]
    
    ## Normalising position data 
    #Subtracting parent joint information from child joint
    
    positions_norm_df = pd.DataFrame()
    
    #Define parent-child joints-pairs
    var_pair_ls = [['BodyCenter'], 
                   ['Chest', 'BodyCenter'],  ['Crotch', 'BodyCenter'],
                   ['HeadJoint', 'Chest'],   ['HeadCenter', 'HeadJoint'],
                   ['RUArmStart', 'Chest'],  ['RLArmStart', 'RUArmStart'], ['RHandStart', 'RLArmStart'],
                   ['LUArmStart', 'Chest'],  ['LLArmStart', 'LUArmStart'], ['LHandStart', 'LLArmStart'],
                   ['RULegStart', 'Crotch'], ['RLLegStart', 'RULegStart'], ['RFootStart', 'RLLegStart'], ['RToesStart', 'RFootStart'],
                   ['LULegStart', 'Crotch'], ['LLLegStart', 'LULegStart'], ['LFootStart', 'LLLegStart'], ['LToesStart', 'LFootStart']]
    
    #Loop though all joint-pairs
    for var_pair in var_pair_ls:        
        #Loop through each joint 3 times = each axis
        for num in range(1,4):
            var_num = [v + str(num) for v in var_pair]            
            if len(var_pair) > 1:
                positions_norm_df[var_num[0]] = data_df[var_num[0]] - data_df[var_num[1]]     #Subtract parent-joint from child-joint
            else: #'BodyCenter'
                positions_norm_df[var_num[0]] = data_df[var_num[0]]   
    
    
    ## Calculate area of movement / recording
    vol_df = pd.DataFrame(columns = var_ls)
    
    for var in var_ls:
        if var != 'BodyCenter':
            #Extract data
            data = positions_norm_df.filter(regex = '^' + var).values
            
            #Compute the convex hull
            hull = ConvexHull(data)
            vol_df[var] = [hull.volume]
        else:
            vol_df[var] = [0]
            
    return(vol_df)

#%% Summary statistics function

#Example
#disp_w_dt = feature_stat_func (disp_f_df, positions_var, init_dt)

def feature_stat_func (data_df, var_ls, init_dt):
    
    """ This function calculates summary statistics over features within each window.
    The measures are saved in a dictionary """
    
    #Initiate
    n_timestep = init_dt['n_timestep']                #Number of timesteps in the recording
    n_frame  = init_dt['n_frame']                     #Number of frames in the data [timesteps]
    n_window = init_dt['n_window']                    #Number of windows in the data [frames]
    frame    = init_dt['frame_width']
    window   = init_dt['window_width']                #Window width [frames]
    overlap  = init_dt['overlap']                     #Overlap width [frames]
    
    #all_df = pd.DataFrame()
    mean_w_df = pd.DataFrame(columns = var_ls)
    var_w_df = pd.DataFrame(columns = var_ls)
    min_w_df = pd.DataFrame(columns = var_ls)
    max_w_df = pd.DataFrame(columns = var_ls)
    entro_w_df = pd.DataFrame(columns = var_ls)

    for var in var_ls: 
        
        data = data_df[var]
        data = data[~np.isnan(data)]
        
        #Initialise
        mean_w_vec  = np.full(n_window, np.nan)
        var_w_vec   = np.full(n_window, np.nan)
        min_w_vec   = np.full(n_window, np.nan)
        max_w_vec   = np.full(n_window, np.nan)
        entro_w_vec = np.full(n_window, np.nan)
        
        #Make an index of "window-overlap"-positions in time
        window_index = [i for i in range(window, n_frame, window-overlap)]
            
        #Looping through the windows to calcualte measure within each window
        for i in range(len(window_index)):
            window_i = window_index[i]      #the position in data of the i-th window
            data_w = data[window_i - window : window_i]
            
            #Calculate statistics of the frame-features between the i'th and the i'th-1 window
            if len(data_w) > 1 :
                mean_w_vec[i] = np.mean(data_w)
                var_w_vec[i]  = stat.variance(data_w)
                min_w_vec[i]  = np.min(data_w)
                max_w_vec[i]  = np.max(data_w)
                
            #Calculate entropy
            if np.mean(data_w) == 0.0:
                entro_w_vec[i] = np.nan
            
            else:
                #Use an exponential distribution
                lambda_estimate = 1 / np.mean(data_w)                                #Calculate rate parameter
                exponential_dist = stats.expon(scale = 1/lambda_estimate)   
                x_values = np.linspace(0, 10 * np.mean(data_w), len(data_w))         #Using 10*np.mean(data) as a 'synthetic' inf = upper bound 
                p_x = exponential_dist.pdf(x_values)
                integrand = -p_x * np.log(p_x) 
                entropy = np.trapz(integrand, x_values)
                
                entro_w_vec[i] = entropy
                
        ##Arrange measures in dataframes
        mean_w_df[var] = mean_w_vec
        var_w_df[var] = var_w_vec
        min_w_df[var] = min_w_vec
        max_w_df[var] = max_w_vec
        entro_w_df[var] = entro_w_vec
        
    all_dt = {'mean': mean_w_df, 'var': var_w_df, 'min': min_w_df, 'max': max_w_df, 'entro': entro_w_df}
        
    return(all_dt)

#%%#########################
# FUNCTIONS FOR REGRESSION #
############################
#%% Regression function

#Example
#Reg_func (X_flat, y, n_splits = 5, model = 'Linear', model_parameter = None) 

def Reg_func (X_flat, y, n_splits, model, model_parameter):
    
    """ This function predicts y from X_flat.
    The function can use different regression models defined in model and model_parameter"""
    
    # Initialize KFold cross-validation
    k_fold = KFold(n_splits = 5, shuffle = True, random_state = 42)
    
    #Initialise
    mse_ls = []
    y_pred_all = []
    y_test_all = []
    
    if model == 'KNN':
        k = model_parameter
        regressor = KNeighborsRegressor(n_neighbors = k)
    elif model == 'SVM':
        kernel = model_parameter
        regressor = SVR(kernel = kernel)
    elif model == 'Linear':
        regressor = LinearRegression()
        
    #Cross-validation
    for train_index, test_index in k_fold.split(X_flat):
        X_train, X_test = X_flat[train_index], X_flat[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        #Train
        regressor.fit(X_train, y_train)

        #Predict
        y_pred = regressor.predict(X_test)
        
        # Calculate MSE
        mse = mean_squared_error(y_test, y_pred)
        mse_ls.append(mse)
        
        y_pred_all.append(y_pred)
        y_test_all.append(y_test)
    
    #For each k - print mean accuracy score and confusion matrix
    print(model, " RMSE: ", round(np.sqrt(np.mean(mse_ls)),4), 'R2: ', round(r2_score(y_test_all, y_pred_all), 2))


#%% KNN regression function

#Example
#KNNreg_func (X_flat, y, 5)

def KNNreg_func (X_flat, y, n_splits):
    
    """ This function predicts y from X_flat.
    The function uses a K-Nearest Neightbor model
    The function uses a n-split-fold Cross Validation framework"""
    
    # Initialize KFold cross-validation
    k_fold = KFold(n_splits = 5, shuffle = True, random_state = 42)
    #k_fold = StratifiedKFold(n_splits = n_splits, shuffle = True, random_state = 42) #represent all ages in each split
    
    #Initialise
    #mean_accuracy_scores = []
    mean_mse_ls = []
    mse_k_ls = []
    
    #Loop over different k-values
    for k in range(2,30,2):
        #accuracy_scores = []
        mse_ls = []
        
        knn_regressor = KNeighborsRegressor(n_neighbors=k)
        
        #Cross-validation
        for train_index, test_index in k_fold.split(X_flat):
            X_train, X_test = X_flat[train_index], X_flat[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            #Train
            knn_regressor.fit(X_train, y_train)
    
            #Predict
            y_pred = knn_regressor.predict(X_test)
            #y_pred_all.extend(y_pred)
            #y_test_all.extend(y_test)
            
            # Calculate MSE
            mse = mean_squared_error(y_test, y_pred)
            mse_ls.append(mse)
    
        #Calculate mean accuracy score for k
        #mean_accuracy = np.mean(accuracy_scores)
        #mean_accuracy_scores.append(mean_accuracy)
        
        # Calculate mean MSE for k
        mean_mse = np.mean(mse_ls)
        mean_mse_ls.append(mean_mse)
        
        mse_k_ls.append([mean_mse, k])
        
        #For each k - print mean accuracy score and confusion matrix
        #print("Accuracy for k =", k, ":", mean_accuracy)
        print("Mean Squared Error for k =", k, ":", round(mean_mse,4))
    
    #Print the best result
    best = min(mse_k_ls, key=lambda x: x[0])
    print('MSE: ', round(best[0],4), ' RMSE: ', round(np.sqrt(best[0]),4), ' for k = ', best[1])
    
    #Run best model
    y_pred_all = []
    y_test_all = []
    knn_regressor = KNeighborsRegressor(n_neighbors = best[1])
    for train_index, test_index in k_fold.split(X_flat):
        X_train, X_test = X_flat[train_index], X_flat[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        #Train
        knn_regressor.fit(X_train, y_train)

        #Predict
        y_pred = knn_regressor.predict(X_test)
        y_pred_all.extend(y_pred)
        y_test_all.extend(y_test)
        
        # Calculate MSE
        mse = mean_squared_error(y_test, y_pred)
        mse_ls.append(mse)
    
    print("RMSE: ", round(np.sqrt(np.mean(mse_ls)),4), 'R2: ', round(r2_score(y_test_all, y_pred_all), 2))

    
#%% SVM regression function

#Example
#lin_coef = SVMreg_func (X_flat, y, n_splits = 5, kernel = 'linear') #

def SVMreg_func (X_flat, y, n_splits, kernel):
    
    """ This function predicts y from X_flat.
    The function uses a Support Vector Machine model w. defined kernel
    The function uses a n-split-fold Cross Validation framework"""
      
    #Initialize 
    i = 1           #Count the folds
    mse_ls = []     #Store MSE values in each fold
    R2_ls = []      #Store R2 values in each fold
    coef_ls = []    #Store feature-coefficients
    y_pred_all = []
    y_test_all = []
    
    #KFold cross-validation 
    k_fold = KFold(n_splits = n_splits, shuffle = True, random_state = 42)
    
    #SVR regressor
    svr_regressor = SVR(kernel = kernel)
    
    #Cross-validation 
    for train_index, test_index in k_fold.split(X_flat):
        print("Fold ", i,"/",n_splits)
        i += 1
        
        X_train, X_test = X_flat[train_index], X_flat[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        #Train
        svr_regressor.fit(X_train, y_train)

        #Predict
        y_pred = svr_regressor.predict(X_test)
        y_pred_all.extend(y_pred)
        y_test_all.extend(y_test)
        
        #Calculate MSE
        mse = mean_squared_error(y_test, y_pred)
        mse_ls.append(mse)
        
        #Calcualte R2 score
        R2 = r2_score(y_test, y_pred)
        R2_ls.append(R2)
        
        #Feature coefficients:
        if kernel == 'linear':
            coef_ls.append(svr_regressor.coef_)
    
    #Calculate mean MSE and R2 score
    mean_mse = np.mean(mse_ls)
    mean_R2 = np.mean(R2_ls)
    print('RMSE: ', round(np.sqrt(mean_mse),4), ' R2: ', round(mean_R2,4))
    
    #Calculate mean feature-coefficients
    if kernel == 'linear':
        stacked_ls = np.stack(coef_ls, axis = 0)
        mean_coef = np.mean(stacked_ls, axis = 0)
    
        #return(mean_coef)
    
    #Prediction
    #y_pred = svr_regressor.predict(X_flat)
    return(y_pred)
#%% Random forest regression function

#Example
#y_true, y_pred, coef = ForestRegCV_func(X_flat, y, n_splits = 5, param = best_param)

def ForestRegCV_func (X_flat, y, n_splits, param):
    
    """ This function predicts y from X_flat.
    The function uses a Random Forest model w. defined parameters
    The function uses a n-split-fold Cross Validation framework"""
    
    #Initialise
    mse_ls = []
    R2_ls = []
    coef_ls = []
    y_pred_all = [] 
    y_true_all = []
    
    # Initialize KFold cross-validation
    k_fold = KFold(n_splits = n_splits, shuffle = True, random_state = 42)
    #k_fold = StratifiedKFold(n_splits = n_splits, shuffle = True, random_state = 42)
    #k_fold = GroupKFold(n_splits = n_splits)
    
    # Initialise Leave-one-out cross-validation
    #loo = LeaveOneOut()
   
    #Random forest regressor
    rf_regressor = RandomForestRegressor(n_estimators      = param['n_estimators'], 
                                         max_depth         = param['max_depth'], 
                                         min_samples_split = param['min_samples_split'],
                                         min_samples_leaf  = param['min_samples_leaf'],
                                         max_leaf_nodes    = param['max_leaf_nodes'],
                                         max_features      = param['max_features'],
                                         min_impurity_decrease = param['min_impurity_decrease'],
                                         bootstrap         = param['bootstrap'],
                                         max_samples       = param['max_samples'], 
                                         random_state      = 42,
                                         criterion         = param['loss_function'])
    
    for train_index, test_index in k_fold.split(X_flat):
        X_train, X_test = X_flat[train_index], X_flat[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        #Train
        rf_regressor.fit(X_train, y_train)
        #print(np.min(y_train), np.max(y_train))

        #Predict
        y_pred = rf_regressor.predict(X_test)
        y_pred_all.extend(y_pred)
        y_true_all.extend(y_test)
        
        #Calculate MSE
        mse = mean_squared_error(y_test, y_pred)
        mse_ls.append(mse)
        
        #Calcualte R2 score
        #R2 = r2_score(y_test, y_pred)
        #R2_ls.append(R2)
        
        #Feature coefficient:
        coef_ls.append(rf_regressor.feature_importances_)
    
    
    #Return
    y_pred = np.array(y_pred_all)
    y_true = np.array(y_true_all)
    
    R2 = r2_score(y_true, y_pred)
    RMSE = np.sqrt(np.mean(mse_ls))
    print('RMSE: ', round(RMSE,4), ' R2: ', round(R2,4))
    #print('RMSE: ', round(np.sqrt(np.mean(mse_ls)),4))
    
    #Calculate mean feature coefficient
    #stacked_ls = np.stack(coef_ls, axis = 0)
    #coef = np.mean(stacked_ls, axis = 0)
    coef = np.mean(coef_ls, axis = 0)

    #y_pred = rf_regressor.predict(X_flat)
    #coef = rf_regressor.feature_importances_
    return(y_true_all, y_pred_all, coef)

#%% Pair features function (for backward selection)

#Example
#feature_name_ls, feature_index = feature_list_func (colnames)

def feature_list_func (X_colnames):
    
    """ This function pairs features based on their columnnames """

    #Group features of anatomic-related joints
    feature_name_ls = []
    index = []
    i = 0
    
    while i < len(X_colnames):
        feature_i = X_colnames[i]
        if i < len(X_colnames)-1:
            feature_1 = X_colnames[i+1]
        
        if 'Body' in feature_i or 'Head' in feature_i or 'cluster' in feature_i:
            feature_name_ls += [feature_i]
            index += [[i]]
            i += 1
            
        elif 'RHand' in feature_i and 'LHand' in feature_1:
            feature_name_ls += [feature_i[:-5] + 'RHand']
            feature_name_ls += [feature_i[:-5] + 'LHand']
            
            index += [[i, i + 1]]
            i += 2
        
        elif 'RFoot' in feature_i and 'LFoot' in feature_1:
            feature_name_ls += [feature_i[:-5] + 'RFoot']
            feature_name_ls += [feature_i[:-5] + 'LFoot']
            
            index += [[i, i + 1]]
            i += 2

        elif 'RLArm' in feature_i or 'LLArm' in feature_1:
            feature_name_ls += [feature_i[:-10] + 'RLArm']
            feature_name_ls += [feature_i[:-10] + 'LLArm']
            
            index += [[i, i + 1]]
            i += 2
            
        elif 'RLLeg' in feature_i or 'LLLeg' in feature_1:
            feature_name_ls += [feature_i[:-10] + 'RLLeg']
            feature_name_ls += [feature_i[:-10] + 'LLLeg']
            
            index += [[i, i + 1]]
            i += 2
                
    return(feature_name_ls, index)

#%% Random forest regression (Backward selection)

#Example
#result_df, plot_mat = Forest_backward_selection_func (X_flat, y, 5, best_param, feature_name_ls, feature_index, version)

def Forest_backward_selection_func (X_flat, y, n_splits, param, feature_name_ls, feature_index, version):
    
    """ This function rund a Random Forest model w. defined parameters
    within a Backward Selection framework based on paired features from feature_name_ls"""
    
    #Define KFold cross-validation 
    k_fold = KFold(n_splits = n_splits, shuffle = True, random_state = 42) 

    #Define Random forest regressor
    rf_regressor = RandomForestRegressor(n_estimators      = param['n_estimators'], 
                                         max_depth         = param['max_depth'], 
                                         min_samples_split = param['min_samples_split'],
                                         min_samples_leaf  = param['min_samples_leaf'],
                                         max_leaf_nodes    = param['max_leaf_nodes'],
                                         max_features      = param['max_features'],
                                         min_impurity_decrease = param['min_impurity_decrease'],
                                         bootstrap         = param['bootstrap'],
                                         max_samples       = param['max_samples'], 
                                         random_state      = 42,
                                         criterion         = param['loss_function'])

    #Define initial dimensions
    n_patients = 125
    n_cluster = 1
    n_volume = 6
    n_dist = 6
    n_special = n_cluster + n_volume + n_dist
    n_window = 210
    n_features = n_window + n_special
    n_feature_pairs = len(feature_index)
    
    ##Reshape X
    #Extract window-independent features
    X_special = X_flat[:,-n_special:]
    X_flat = X_flat[:,:-n_special] #delete dist 
    #Extract window-features
    X_window = X_flat.reshape(125, 1083, n_window) #125 patients, 1083 inteprolated windows, n features)
   
    feature_ls = [i for i in range(n_features)] 
    feature_window = [i for i in range(n_window)] 
    feature_special = [i for i in range(n_special)] 
    measure_selection = pd.DataFrame()
    run = 0

    #Print and save
    path_file = os.path.join(path_model, 'Backward selection/BS1_' + version +'.txt')
    with open(path_file, 'w') as file:
        file.write('Iteration:  Feature removed:  Prediction range [weeks]:  R^2:  RMSE: ')
    
    print('Iteration: ', 'Feature removed: ', 'Prediction range [weeks]: ', 'R^2: ', 'RMSE: ')

    # SELECTION-LOOP #
        #Iterate until the model contain 1 feature
        #Remove one feature permanently in each iteration
        
        
    y_true_all, y_pred_all, R2, RMSE = ForestRegCV1_func (X_flat, y, n_splits, param)
    R2_trial = R2.copy()
    
    plot_mat = np.zeros((n_feature_pairs, n_patients))
    while R2 >= R2_trial:
        
        start_time = time.time()
        
        #Initialise "trial-meausures"
        measure_trial = []
        
        # TRIAL-LOOP #
            #Loop though all remaining features - remove one temporarily in each iteration
        for index in feature_index:
            
            #Initialise fold-measures
            R2_fold = []
            MSE_fold = []
            y_pred_fold = []
            y_true_fold = []
            
            ##Decide which feature to remove temporarily
            #index refers to a window-dependent feature
            if run != 0:
                if index[0] < n_window: 
                    keep_index = feature_window.copy()
                    
                    if len(index) == 1: #single feature
                        keep_index.remove(index[0])
                        
                    elif len(index) == 2: #grouped feature
                        keep_index.remove(index[0])
                        keep_index.remove(index[1])
                        
                    X_window_keep = X_window[:, :, keep_index]
                    X_special_keep = X_special.copy()
                    
                #index refers to a special features
                elif index[0] > n_window-1 and index[0] < n_window + n_special: 
                    keep_index = feature_special.copy()
                    
                    if len(index) == 1: #single feature
                        index_1 = index[0] - n_window
                        keep_index.remove(index_1)
                        
                    elif len(index) == 2: #grouped feature
                        index_1 = index[0] - n_window
                        index_2 = index[1] - n_window
                        keep_index.remove(index_1)
                        keep_index.remove(index_2)
                        
                    X_special_keep = X_special[:, keep_index]
                    X_window_keep = X_window.copy() 
            
            ##Flatten X_window and combine with special features into X subset
            X_flat = X_window_keep.reshape(X_window_keep.shape[0], -1)
            X_subset = np.hstack((X_flat, X_special_keep))

            #Cross-validation
            for train_index, test_index in k_fold.split(X_subset):
                X_train, X_test = X_subset[train_index], X_subset[test_index]
                y_train, y_test = y[train_index], y[test_index]
                
                #Train
                rf_regressor.fit(X_train, y_train)

                #Predict
                y_pred = rf_regressor.predict(X_test)
                y_pred_fold.append(y_pred)
                y_true_fold.append(y_test)
                
                #Calculate R2
                R2 = r2_score(y_test, y_pred)
                R2_fold.append(R2)
                
                #Calculate MSE
                MSE = mean_squared_error(y_test, y_pred)
                MSE_fold.append(MSE)
                
                #Calculate prediction-range
                #pred_range = np.max(y_pred) - np.min(y_pred)
                #range_fold.append(pred_range)
                
            #Final prediction:
            #Store performance measures for model when i'th feature is removed            
            MSE_trial = np.mean(MSE_fold)
            R2_trial  = np.mean(R2_fold)
            range_trial  = np.max(y_pred) - np.min(y_pred)
            measure_trial.append((MSE_trial, range_trial, R2_trial, index))
            #measure_trial.append((range_trial, MSE_trial, R2_trial, index))
            
           
        #Evaluate the performance of each iteration of the trial-loop
        measure_trial.sort(reverse=True)
        #MSE_selection, range_selection, R2_selection, feature_remove_index = measure_trial[0]
        MSE_selection, range_selection, R2_selection, feature_remove_index = measure_trial[0]
        
        #Remove the feature that, when removed, gives the best performance
        if len(feature_remove_index) == 1: #single feature
            feature_remove_name = [feature_name_ls[feature_remove_index[0]]]
            feature_name_ls = feature_name_ls[:feature_remove_index[0]] + feature_name_ls[feature_remove_index[0]+1:]
            feature_index.remove(feature_remove_index)
            if feature_remove_index[0] < n_window: 
                feature_window.remove(feature_remove_index[0])
            elif feature_remove_index[0] > n_window-1 and feature_remove_index[0] < n_window + n_special: 
                feature_special.remove(feature_remove_index[0] - n_window)
        elif len(feature_remove_index) == 2: #grouped feature
            feature_remove_name = [feature_name_ls[feature_remove_index[0]], feature_name_ls[feature_remove_index[1]]]
            feature_name_ls = feature_name_ls[:feature_remove_index[0]] + feature_name_ls[feature_remove_index[1]+1:]
            feature_index.remove(feature_remove_index)
            if feature_remove_index[0] < n_window: 
                feature_window.remove(feature_remove_index[0])
                feature_window.remove(feature_remove_index[1])
            elif feature_remove_index[0] > n_window-1 and feature_remove_index[0] < n_window + n_special: 
                feature_special.remove(feature_remove_index[0] - n_window)
                feature_special.remove(feature_remove_index[1] - n_window)
        
        #Store and print
        measure_selection[run] = [feature_remove_name, range_selection, np.sqrt(MSE_selection), R2_selection]
        print(run, feature_remove_name, round(range_selection,2), round(R2_selection,2), round(np.sqrt(MSE_selection),2))
        
        #path_file = os.path.join(path_model, 'Backward selection/BS_100524.txt')
        with open(path_file, 'a') as file:
            line = f"{run}, {feature_remove_name}, {round(range_selection, 2)}, {round(R2_selection, 2)}, {round(np.sqrt(MSE_selection), 2)}\n"
            file.write(line)
        
        #Plot predictions
        y_true_fold = np.concatenate(y_true_fold)
        y_pred_fold = np.concatenate(y_pred_fold) 
        
        coefficients = np.polyfit(y_true_fold, y_pred_fold, 1)  # Linear fit (degree=1)
        poly = np.poly1d(coefficients)
        trendline = poly(y_true_fold)
        plot_mat[run] = trendline
        
        fig = plt.figure(figsize = (12, 9))
        plt.grid(axis = 'y', alpha = 0.75)
        plt.title(str(run) + ' Prediction of age', fontsize = 20) #20 = big
        plt.scatter(y_true_fold, y_pred_fold, color = DTU_color[2])
        plt.plot(y_true_fold, trendline, linestyle='--', linewidth=1, label='Trend line', color = DTU_color[0])
        plt.xlabel('True')
        plt.ylabel('Pred')
        plt.xlim(min(y_true_fold), max(y_true_fold))
        plt.ylim(min(y_true_fold), max(y_true_fold))
        plt.legend()
        plt.show() 
        #path_plot = os.path.join(path_model, 'Backward selection/' + version + '_' + str(run) +'.png')
        #plt.savefig(path_plot) 
                
        #Update run and run-time
        run += 1  
        
        run_time = time.time()-start_time
        print('(Time: ', round(run_time/60,2), ' min)')
        
    return(measure_selection, plot_mat)
#%% Random forest regression function (Hold-out)

#Example
#y_true_ML, y_pred_ML, y_true_HO, y_pred_HO =  ForestRegHoldout_func (X_flat, y, 5, best_param, X_holdout, y_holdout)

def ForestRegHoldout_func (X_flat, y, n_splits, param, X_holdout, y_holdout):
    
    """ This function predicts y from X_flat based on hold-out data.
    The function filts a Random Forest model w. defined parameters on X_flat and y 
    The function uses the model to predict y_holdout from X_holdout
    The function uses a n-split-fold Cross Validation framework"""
    
    #Initialise
    mse_ls = []
    R2_ls = []
    coef_ls = []
    y_pred_all = [] 
    y_true_all = []
    
    # Initialize KFold cross-validation
    k_fold = KFold(n_splits = n_splits, shuffle = True, random_state = 42)
   
    #Random forest regressor
    rf_regressor = RandomForestRegressor(n_estimators      = param['n_estimators'], 
                                         max_depth         = param['max_depth'], 
                                         min_samples_split = param['min_samples_split'],
                                         min_samples_leaf  = param['min_samples_leaf'],
                                         max_leaf_nodes    = param['max_leaf_nodes'],
                                         max_features      = param['max_features'],
                                         min_impurity_decrease = param['min_impurity_decrease'],
                                         bootstrap         = param['bootstrap'],
                                         max_samples       = param['max_samples'], 
                                         random_state      = 42,
                                         criterion         = param['loss_function'])
    
    for train_index, test_index in k_fold.split(X_flat):
        X_train, X_test = X_flat[train_index], X_flat[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        #Train
        rf_regressor.fit(X_train, y_train)

        #Predict
        y_pred = rf_regressor.predict(X_test)
        y_pred_all.extend(y_pred)
        y_true_all.extend(y_test)
        
        #Calculate MSE
        mse = mean_squared_error(y_test, y_pred)
        mse_ls.append(mse)
        
        #Feature coefficient:
        coef_ls.append(rf_regressor.feature_importances_)
    
    #Evaluate training
    y_pred = np.array(y_pred_all)
    y_true = np.array(y_true_all)
    
    R2 = r2_score(y_true, y_pred)
    print('RMSE: ', round(np.sqrt(np.mean(mse_ls)),4), ' R2: ', round(R2,4))
    
    ##Train final model on the entire training set
    rf_regressor.fit(X_flat, y)
    
    #Evaluate on holdout data
    y_holdout_pred = rf_regressor.predict(X_holdout)
    holdout_mse = mean_squared_error(y_holdout, y_holdout_pred)
    holdout_r2 = r2_score(y_holdout, y_holdout_pred)
    
    print('RMSE (Holdout): ', round(np.sqrt(holdout_mse), 4), ' R2 (Holdout): ', round(holdout_r2, 4))

    return(y_true_all, y_pred_all, y_holdout, y_holdout_pred)




#%%###########################
# FUNCTIONS FOR SEGMENTATION #
##############################
#%% PCA function

#Example
#V, S = PCA_func (frame_df)

def PCA_func (data_df):
    
    """ This function calcualtes SVD """
    
    #SVD decomposition
    # D = USV^T
    #U, S, Vt = np.linalg.svd(data_df)
    #V = Vt.T
    #U = orthonomal matrix = orthonormal vectors u
    #S = Sigma = diagonal matrix = singular values (eigenvalue) in diagonal = how much variability each PC descirbe
    #V = orthonomal matrix = orthonormal vectors v = PC's (eigenvectors)
    
    ##Use PCA for SVD decomposition (to get rid of the large U-matrix)
    #Compute the covariance matrix from the centered data
    cov_mat = np.cov(data_df, rowvar = False)
    
    #Compute eigenvalues and eigenvectors from covariance matrix
    eigen_val, eigen_vec = np.linalg.eigh(cov_mat) #using the sample covariance = need to scale eigen values to become singular calues
    
    #Sort eigenvalues and -vectors according to descending eigenvalues
    sort_index = np.argsort(eigen_val)[::-1]
    Vt = eigen_vec[:, sort_index]
    V = Vt.T
    num_samples = data_df.shape[0]
    
    with np.errstate(invalid='ignore'): #Suppress the RuntimeWarning
        S = np.sqrt(eigen_val[sort_index] * num_samples) #Sort and scale eigenvalues to become singular values (scale by sqr and num of samples)
    
    return(V, S)

#%% Gaussian distribution

#Example
#C = Gaussian_func (S, V, r, n_axis, n_frame)

def Gaussian_func (S, V, r, n_axis):
    
    """ This function approximates a Gaussian distribution to data """
    
    #Compute singular values from S - full size
    sing_val_r = S[:r] 
    sing_val_mean = np.array([np.nanmean(S[r+1:]) for i in range(n_axis-r)])    #replacing discarded singular values with their mean 
    sing_val = np.concatenate((sing_val_r, sing_val_mean))
    
    I = np.eye(n_axis)
    Sigma = np.dot(sing_val, I)
    Lambda = np.diag(sing_val ** 2) #Sigma.T * Sigma
    
    Cov = 1/n_frame * np.dot(V, np.dot(Lambda, V.T))
    
    return(Cov)

#%% Mahalanobis distance function

#Example
#H = Mahalanobis_func (angles_center_df, C, K, T)

def Mahalanobis_func (data_df, C, K, T):
    
    """ This function calculates the mahalanobis distance
    and thus determines how likely the next data points is to belong to a Gaussian distribution """
    
    C_inv = np.linalg.inv(C) #Covariance matrix of the Gaussian distribution of the first k frames
    data_KT = data_df.iloc[K+1 : K+T]
    data_1K = data_df.iloc[:K]
    
    H1 = []
    for i in range(K+1, K+T):
        data = data_df.iloc[i]
        #diff = data - np.mean(data_KT, axis = 0) #should loop over i values instead of being locked at i=0
        diff = data - np.mean(data_1K, axis = 0)
        H1.append(np.dot(np.dot(diff.T, C_inv), diff))
    
    H = np.mean(H1)

    return(H)

#%%#########################
# FUNCTIONS FOR CLUSTERING #
############################
#%% Segmentation + interpolation function

#Example
#data_segment_ls, interpol_segment_ls = segment_interpolation_func (angles_df, segment_index, 5, int(percentile_99)) #13249 segments, 13249 interpolated segments

def segment_interpolation_func (data_df, cut_index, threshold, n_interpol_frame):
    
    """This function segments data according to cut_index
    and interpolated the data according to the threshold"""
    
    #Threshold = the minimum number of rows in a data segment fore interpolation
    
    #Initialise
    data_segment_ls = []
    interpol_segment_ls = []
    
    #Loop though cuts in cut_index
    for i in range(len(cut_index)-1):
        
        cut1 = int(cut_index[i])
        cut2 = int(cut_index[i+1])
        
        ##Cut data into segments
        data_segment = data_df.iloc[cut1:cut2].values
        
        #Error-handling
        if len(data_segment) == 0:
            print(cut1,cut2)
            print(data_segment)
        
        #Save data segment
        data_segment_ls.append(data_segment)
        
        ##Interpolation
        n_segment_frame = data_segment.shape[0]
        
        if n_segment_frame >= threshold:
            
            #Create interpolation function
            x = np.arange(n_segment_frame) #Generate an array of index from 0 to n_segment_frame-1
            interpol_func = interp1d(x, data_segment, axis=0, kind = 'linear') #axis = 0 == interpolate along first axis of data_segment (rows)
            
            #Interpolate segment
            x_interpol = np.linspace(0, n_segment_frame - 1, n_interpol_frame) #Generate n_interpol_frame evenly spread numbers in the interval [0; n_segment_frame - 1]
            interpol_segment = interpol_func(x_interpol)
            interpol_segment_ls.append(interpol_segment)
        
    return(data_segment_ls, interpol_segment_ls)