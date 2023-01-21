# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 11:48:26 2022

@author: fwoff

This script runs the data augmentation process. the file
'LR_processed_data.csv' should be in the same file as this, as that file
cotains the preprocessed data drawn from the dataset.
With no arguements, or a single numerical argument, the script generates a 
new dataset, plots out the old and new datasets, and asks the user if they 
want to save it after the plots are closed. The numerical argument is a scaling
factor for the noise of the GPS data. If the argument "view" is given, 
followed by a path, the script will display information about the dataset. 
Otherwise, if an argument that is not view is given, it will create a folder
with that name and fill it with multiple augmented datasets. If a numerical 
argument is given after that folder, it will be used as a scale for the 
intensity of noise used to generate the GPS data.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tsaug as ts
import random
import sys
import os

"""
Functions used for data augmentation
"""
def TranslateCoordinates(coordList):
    xAdd = random.uniform(-3000.0, 3000.0)
    yAdd = random.uniform(-3000.0, 3000.0)
    #we copy the values and add to all of them to create the new coordinate set
    newList = np.copy(coordList)
    newList[:, 0] = newList[:, 0] + xAdd
    newList[:, 1] = newList[:, 1] + yAdd
    return newList #we return the entirety of the list we made

def rotateCoordinateSet(coords):
    #we extract the first coordinate of the set as the origin, which we will rotate around
    origin = coords[0, :]
    angle = np.random.randint(0, 360)
    angle = np.deg2rad(angle)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    o = np.atleast_2d(origin)
    p = np.atleast_2d(coords)
    return np.squeeze((R @ (p.T-o.T) + o.T).T)

#function for translation
def TranslateAugmentation(df):
  #create the new dataframe that we will return
  new_df = df.copy(deep = True)
  #extract the ground truth values as a numpy array,
  #pass it through the tranformation function, and convert 
  #the output to a dataframe
  gtValuesFrame = df[["x_gt", "y_gt"]]
  gtArray = gtValuesFrame.to_numpy()
  new_gt = TranslateCoordinates(gtArray)
  new_gt_frame = pd.DataFrame(new_gt, columns = ["x_gt","y_gt"])
  #apply the dataframe to the output, and return
  new_df["x_gt"] = new_gt_frame["x_gt"]
  new_df["y_gt"] = new_gt_frame["y_gt"]
  return new_df

#function for rotation
def RotationAugmentation(df):
  #create the new dataframe that we will return
  new_df = df.copy(deep = True)
  #extract the ground truth values as a numpy array,
  #pass it through the tranformation function, and convert 
  #the output to a dataframe
  gtValuesFrame = df[["x_gt", "y_gt"]]
  gtArray = gtValuesFrame.to_numpy()
  new_gt = rotateCoordinateSet(gtArray)
  new_gt_frame = pd.DataFrame(new_gt, columns = ["x_gt","y_gt"])
  #apply the dataframe to the output, and return
  new_df["x_gt"] = new_gt_frame["x_gt"]
  new_df["y_gt"] = new_gt_frame["y_gt"]
  return new_df

#function for time warp
def TimeWarpAugmentation(df):
  new_df = df.copy(deep = True)
  warp_number = random.randint(1, 12)
  gtValuesFrame = df[["x_gt", "y_gt"]]
  gtArray = gtValuesFrame.to_numpy()
  #Where T is the number of elements in each series, and N is the
  #number of series', tsaug's augmentation takes in an array of
  #shape (N,T), while our data is organized columnwise (T, N),
  #so we reshape before and after applying noise
  T, N = gtArray.shape
  gtArray.reshape((N, T))
  ts.TimeWarp(n_speed_change=warp_number).augment(gtArray)
  gtArray.reshape((T, N))
  new_gt_frame = pd.DataFrame(gtArray, columns = ["x_gt","y_gt"])
  new_df["x_gt"] = new_gt_frame["x_gt"]
  new_df["y_gt"] = new_gt_frame["y_gt"]
  return new_df

#function to get acceleration data, that would be measured by an IMU
def GetIMU(df):
  new_df = df.copy(deep = True)
  gt_x_frame = df[["x_gt"]]
  gt_y_frame = df[["y_gt"]]
  gtx = gt_x_frame.to_numpy()
  gty = gt_y_frame.to_numpy()
  #double derivarives for accelerometer data
  acc_x = np.diff(gtx, n=2, axis=0)
  acc_y = np.diff(gty, n=2, axis=0)
  #getting the angular velocity by taking the derivative of the array of angles
  fractions = gty / gtx
  angles = np.arctan(fractions)
  ang_v = np.diff(angles, axis=0)
  #since the derivative for the first two instances don't have accelerometer
  #data, we remove them, as well as the first angular velocity
  #instance, then add the data
  gtValuesFrame = df[["x_gt", "y_gt"]]
  gtArray = gtValuesFrame.to_numpy()
  new_gt_frame = pd.DataFrame(gtArray[2:,:], columns = ["x_gt","y_gt"])
  new_df["x_gt"] = new_gt_frame["x_gt"]
  new_df["y_gt"] = new_gt_frame["y_gt"]
  final_ang_v = ang_v[1:]
  final_ang_v = np.reshape(final_ang_v,(final_ang_v.size, 1))
  new_x_frame = pd.DataFrame(acc_x, columns = ["acc_x"])
  new_y_frame = pd.DataFrame(acc_y, columns = ["acc_y"])
  new_ang_frame = pd.DataFrame(final_ang_v, columns = ["ang_v"])
  new_df["x_acc_true"] = new_x_frame["acc_x"]
  new_df["y_acc_true"] = new_y_frame["acc_y"]
  new_df["ang_v"] = new_ang_frame["ang_v"]
  return new_df


#function to derive sensor data through noise
#In Progress
def SensorData(df, std, scaling=1/2):
  #create the new dataframe that we will return
  new_df = df.copy(deep = True)
  #extract the ground truth values and true acceleration
  #values as a numpy array
  gtValuesFrame = df[["x_gt", "y_gt"]]
  array = gtValuesFrame.to_numpy()
  #Where T is the number of elements in each series, and N is the
  #number of series', tsaug's augmentation takes in an array of
  #shape (N,T), while our data is organized columnwise (T, N),
  #so we reshape before and after applying noise
  T, N = array.shape
  array.reshape((N, T))
  #by default, AddNoise calculates 0 mean gaussian noise (white noise)
  #independantly for each series.
  gt_sensor_readings = ts.AddNoise(scale=(std*scaling), normalize=False).augment(array)
  gt_sensor_readings.reshape((T, N))
  new_gt_frame = pd.DataFrame(gt_sensor_readings, columns = ["x_gps","y_gps"])
  #apply the dataframe to the output, and return
  new_df["x_gps"] = new_gt_frame["x_gps"]
  new_df["y_gps"] = new_gt_frame["y_gps"]
  imuValuesFrame = df[["x_acc_true", "y_acc_true"]]
  array = imuValuesFrame.to_numpy()
  T, N = array.shape
  array.reshape((N, T))
  acc_sensor_readings = ts.AddNoise().augment(array)
  acc_sensor_readings.reshape((T, N))
  new_acc_frame = pd.DataFrame(acc_sensor_readings, columns = ["IMU_x","IMU_y"])
  new_df["IMU_x"] = new_acc_frame["IMU_x"]
  new_df["IMU_y"] = new_acc_frame["IMU_y"]
  #similar process for angular velocity
  gyroValuesFrame = df[["ang_v"]]
  array = gyroValuesFrame.to_numpy()
  T, N = array.shape
  array.reshape((N, T))
  gyro_sensor_readings = ts.AddNoise().augment(array)
  gyro_sensor_readings.reshape((T, N))
  new_gyro_frame = pd.DataFrame(gyro_sensor_readings, columns = ["gyro"])
  new_df["gyro"] = new_gyro_frame["gyro"]
  return new_df

def GetAugmentedDataset(df, scale=1/2, translate = False, rotate = False, warp = False, save = False, file_name = ''):
  new_df = df.copy(deep = True)
  #calculating the standard deviation of the error of the sensors from the dataset
  difference_x = df[["x_gt"]].to_numpy() - df[["translated_onboard_x"]].to_numpy()
  difference_y = df[["y_gt"]].to_numpy() - df[["translated_onboard_y"]].to_numpy()
  std_x = np.std(difference_x)
  std_y = np.std(difference_y)
  std = (std_x + std_y)/2
  
  
  #apply whatever data tranformations have been selected
  if(translate):
    new_df = TranslateAugmentation(new_df)
  if(rotate):
    new_df = RotationAugmentation(new_df)
  if(warp):
    new_df = TimeWarpAugmentation(new_df)
  
  #calculate simulated IMU and sensor data
  new_df = GetIMU(new_df)
  new_df = SensorData(new_df, std, scaling=scale)

  #get rid of empty rows from previous step
  new_df['x_gt'].replace('', np.nan, inplace=True)
  new_df.dropna(subset=['x_gt'], inplace=True)

  #if the new dataset is meant to be saved somewhere, do so, then return
  if(save and (file_name != '')):
    new_df.to_csv(file_name)
  return new_df

def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False

# read data from .csv to dataframe
df = pd.read_csv('LR_processed_data.csv')
scaling = 1/2
makeWithScale = False
if len(sys.argv) == 2:
    if isfloat(sys.argv[1]):
        scaling = float(sys.argv[1])
        makeWithScale = True
elif len(sys.argv) > 2:
    if isfloat(sys.argv[2]):
        scaling = float(sys.argv[2])
        
if len(sys.argv) == 1 or makeWithScale:
    # Create plot from df
    df.plot( x = 'translated_onboard_x', y = 'translated_onboard_y', kind = 'line', label = 'GPS')
    df.plot( x = 'x_gt', y = 'y_gt', kind = 'line', label = 'Ground Truth')
    df_new = GetAugmentedDataset(df, translate=True, rotate=True, warp=True, scale=scaling)
    df_new.plot( x = 'x_gps', y = 'y_gps', kind = 'line', label = 'New GPS Readings')
    df_new.plot( x = 'x_gt', y = 'y_gt', kind = 'line', label = 'New Ground Truth')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    saveState = " "
    while saveState != "n" and saveState != "y":
        saveState = input("Do you want to save the new dataset? (y/n) ")
    if saveState == "y":
        newName = input("Type the file name, leaving out the .csv ") + ".csv"
        df_new.to_csv(newName)
elif sys.argv[1] != "view":
    if not os.path.exists(sys.argv[1]):
        os.makedirs(sys.argv[1] + "/Training_Data")
        os.makedirs(sys.argv[1] + "/Validation_Data")
    trainingBase = sys.argv[1] + "/Training_Data/Training_Set_"
    validationBase = sys.argv[1] + "/Validation_Data/Validation_Set_"
    for i in range(24):
        if i > 13:
            trainingName = trainingBase + str(i) + ".csv"
            newData = GetAugmentedDataset(df, scale=scaling, translate=True, rotate=True, warp=True, save=True, file_name=trainingName)
    for j in range(14):
        if j > 6:
            validationName = validationBase + str(j) + ".csv"
            newData = GetAugmentedDataset(df, translate=True, rotate=True, warp=True, save=True, file_name=validationName)
elif len(sys.argv) >= 3:
    path = sys.argv[2]
    df = pd.read_csv(path)
    df.plot( x = 'x_gt', y = 'y_gt', kind = 'line', label = 'Ground Truth')
    df.plot( x = 'x_gps', y = 'y_gps', kind = 'line', label = 'GPS Readings')
    fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1)
    ax1.plot(df['Unnamed: 0'], df['x_acc_true'], label = 'True X Acceleration', color = 'red')
    plt.xlabel('Time Step')
    plt.ylabel('Acceleration')
    ax2.plot(df['Unnamed: 0'], df['y_acc_true'], label = 'True Y Acceleration', color = 'blue')
    plt.xlabel('Time Step')
    plt.ylabel('Acceleration')
    ax3.plot(df['Unnamed: 0'], df['ang_v'], label = 'True Angular Acceleration', color = 'blue')
    plt.xlabel('Time Step')
    plt.ylabel('Velocity')
    ax4.plot(df['Unnamed: 0'], df['IMU_x'], label = 'IMU X Readings', color = 'yellow')
    plt.xlabel('Time Step')
    plt.ylabel('Acceleration')
    ax5.plot(df['Unnamed: 0'], df['IMU_y'], label = 'IMU Y Readings', color = 'purple')
    plt.xlabel('Time Step')
    plt.ylabel('Acceleration')
    ax6.plot(df['Unnamed: 0'], df['gyro'], label = 'Gyro Readings', color = 'purple')
    plt.xlabel('Time Step')
    plt.ylabel('Velocity')
    plt.show()
    
    


