import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
import matplotlib.pyplot as plt

import sys
import os

'''
  The first argument should be the path to the file. By default this looks at the GPS
  data from the AGL file as the prediction, if the second argument is -m, then
  it compares a model's predictions instead.
'''

def outlierDetection(df_result, model):

  if (model):
      x_gt = torch.tensor(data = df_result['x_gt'], dtype = torch.float)
      y_gt = torch.tensor(data = df_result['y_gt'], dtype = torch.float)
      x_pred = torch.tensor(data = df_result['x_pred'], dtype = torch.float)
      y_pred = torch.tensor(data = df_result['y_pred'], dtype = torch.float)
  else:
      x_gt = torch.tensor(data = df_result[' x_gt'], dtype = torch.float)
      y_gt = torch.tensor(data = df_result[' y_gt'], dtype = torch.float)
      x_pred = torch.tensor(data = df_result[' x_gps'], dtype = torch.float)
      y_pred = torch.tensor(data = df_result[' y_gps'], dtype = torch.float)

  data_gt = torch.stack((x_gt, y_gt), axis = 1)
  data_pred = torch.stack((x_pred, y_pred), axis = 1)

  distance = torch.norm(data_gt - data_pred, dim = 1, p = None) #get distances between corrosponding coordinates
  min_d = distance.min()
  max_d = distance.max()

  hist = torch.histc(distance, bins = 100, min = min_d, max = max_d) #make pytorch histogram 
  bins = 100   #adjustable, must change ^ if changed
  r = range(bins)

  quantiles = torch.tensor([0.75]) #get upper 90% of distance values
  threshold = torch.quantile(distance, quantiles, dim=0, keepdim=True) #set threshold for determining outliers
  
  #quartile threshold for model comparison
  #Distance = 13 for overall accuracy has proven most consistent
  
  plt.figure(figsize=(12, 8))
  plt.title("Detecting Outliers")
  plt.scatter(data_gt[:, 0], data_gt[:, 1], color = 'b') #ground truth points
  for i in range(len(distance)):
      if distance[i] > threshold and distance[i] > 15:
          plt.scatter(data_pred[i][0], data_pred[i][1], color = 'r', s = 20) #values that are outliers
      else:
          plt.scatter(data_pred[i][0], data_pred[i][1], color = 'g', s = 8) #values within max distance


   #if you just want to see the data point distances, uncomment the code below and comment the code below
  '''
  plt.figure(figsize=(12, 8))
	  distances = []
	  if len(df_result['x_gt']) == len(df_result['x_pred']):
		for row in df_result.iterrows():
		  dist = np.sqrt((row[1][0] - row[1][2])**2 + (row[1][1] - row[1][3])**2)
		  distances.append(dist)
		df = pd.DataFrame(distances, columns = ['distance'])
		new_df = df.sort_values(by = 'distance')
		new_df.hist(column='distance', bins=50, grid=False, figsize=(12,8))
  '''
		
  plt.show()


if len(sys.argv) < 2:
    print("Please provide a file path to GroundTruthAGL.csv or a model's predictions")
else:
    df = pd.read_csv(sys.argv[1])
    modelPredict = False
    if len(sys.argv) > 2:
        modelPredict = (sys.argv[2] == "-m")
    outlierDetection(df, modelPredict)