import pandas as pd
import numpy as np
from pyproj import Proj, transform

# This script takes the onBoardGPS values and maps them to groundTruthAGL
# values (while translating from lat, lon to x, y)
def main():

    # read data from .csv to dataframe
    df_agl = pd.read_csv('../datasets/GroundTruthAGL.csv')
    df_onboardGPS = pd.read_csv('../datasets/OnboardGPS.csv')

    new_df = pd.DataFrame()
    new_df['timestep'] = np.zeros(len(df_agl['imgid']))
    new_df['imgid'] = np.zeros(len(df_agl['imgid']))
    new_df['x_gt'] = np.zeros(len(df_agl['imgid']))
    new_df['y_gt'] = np.zeros(len(df_agl['imgid']))
    new_df['onboard_lat'] = np.zeros(len(df_agl['imgid']))
    new_df['onboard_lon'] = np.zeros(len(df_agl['imgid']))
    new_df['translated_onboard_x'] = np.zeros(len(df_agl['imgid']))
    new_df['translated_onboard_y'] = np.zeros(len(df_agl['imgid']))

    new_csv(new_df, df_agl, df_onboardGPS)
    latlon_to_xy(new_df)

    new_df.to_csv('../datasets/LR_processed_data.csv')

# Combines relevant data across different .csv's into one .csv in order to have localized data
def new_csv(new_df, df_agl, df_onboardGPS):
  for idx, elem in enumerate(df_agl['imgid']):
    row_gt = df_agl.loc[df_agl['imgid'] == elem].to_numpy()
    row_onboard = df_onboardGPS.loc[df_onboardGPS[' imgid'] == elem].to_numpy()

    if row_onboard.size > 0 and row_gt.size:
      new_df['x_gt'][idx] = row_gt[0][1]
      new_df['y_gt'][idx] = row_gt[0][2]
      new_df['onboard_lat'][idx] = row_onboard[0][2]
      new_df['onboard_lon'][idx] = row_onboard[0][3]
      new_df['timestep'][idx] = row_onboard[0][0]
      new_df['imgid'][idx] = row_onboard[0][1]
    else:
      new_df['x_gt'][idx] = new_df['x_gt'][idx - 1]               # If missing data use previous row
      new_df['y_gt'][idx] = new_df['y_gt'][idx - 1]
      new_df['onboard_lat'][idx] = new_df['onboard_lat'][idx - 1]
      new_df['onboard_lon'][idx] = new_df['onboard_lon'][idx - 1]
      new_df['timestep'][idx] = new_df['timestep'][idx -1]
      new_df['imgid'][idx] = new_df['imgid'][idx - 1]

# Transform coordinates from longitude, latitude to x, y
def latlon_to_xy(new_df):
  outProj = Proj('epsg:32632') # WGS 84 / UTM zone 32N coordinate system
  inProj = Proj('epsg:4326') # Latitude, longitude

  new_df['translated_onboard_x'] = new_df.apply(lambda row: transform(inProj, outProj, row['onboard_lat'], row['onboard_lon'])[0], axis=1)
  new_df['translated_onboard_y'] = new_df.apply(lambda row: transform(inProj, outProj, row['onboard_lat'], row['onboard_lon'])[1], axis=1)

if __name__ == '__main__':
    main()