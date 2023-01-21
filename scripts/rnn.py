import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import torchmetrics
import torch.optim as optim
import sys

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Note: The RNN functions were taken and built upon from the example provided by Kaan Kuguoglu at towardsdatascience.com
# https://towardsdatascience.com/building-rnn-lstm-and-gru-for-time-series-using-pytorch-a46e5b094e7b

# ------------------------------------------------------------------------------------------------------
# Module and Trainer classes defined below
# ------------------------------------------------------------------------------------------------------

# A class for a simple PyTorch RNN Model
class RNN_Model(nn.Module):
    def __init__(self, input_size, hidden_dim, n_layers, output_size):
        super(RNN_Model, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.hidden_dim = hidden_dim
        self.layer_dim = n_layers

        # RNN layers
        self.rnn = nn.rnn(
            input_size, hidden_dim, n_layers, batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)
    
    def forward(self, x):

        # Initializing hidden state for first input using method defined below
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).requires_grad_()

        # Passing in the input and hidden state into the model and obtaining outputs
        out, h0 = self.rnn(x, h0.detach())
		
        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)
        
        return out


# A class to train a PyTorch RNN model
class Trainer(pl.LightningModule):
    def __init__(self, model, loss_fn, optimizer):
        super(Trainer, self).__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_losses = []
        self.val_losses = []
    
    def train_step(self, x, y):
        # Sets model to train mode
        self.model.train()

        # Makes predictions
        yhat = self.model(x)

        # Computes loss
        loss = self.loss_fn(y, yhat)

        # Computes gradients
        loss.backward()

        # Updates parameters and zeroes gradients
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Returns the loss
        return loss.item()

    def train(self, trial, train_loader, val_loader, batch_size, n_epochs, n_features=1):

        for epoch in range(1, n_epochs + 1):
            batch_losses = []
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.view([batch_size, -1, n_features])
                y_batch = y_batch
                loss = self.train_step(x_batch, y_batch)
                batch_losses.append(loss)
            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)

            with torch.no_grad():
                batch_val_losses = []
                for x_val, y_val in val_loader:
                    x_val = x_val.view([batch_size, -1, n_features])
                    y_val = y_val
                    self.model.eval()
                    yhat = self.model(x_val)
                    val_loss = self.loss_fn(y_val, yhat).item()
                    batch_val_losses.append(val_loss)

                validation_loss = np.mean(batch_val_losses)
                self.val_losses.append(validation_loss)

            if (epoch <= 10) | (epoch % 50 == 0):
                print(
                    f"[{epoch}/{n_epochs}] Training loss: {training_loss:.8f}\t Validation loss: {validation_loss:.8f}"
                )


    def evaluate(self, test_loader, batch_size, n_features):
        with torch.no_grad():
            predictions = []
            values = []
            for x_test, y_test in test_loader:
                x_test = x_test.view([batch_size, -1, n_features])
                y_test = y_test
                self.model.eval()
                yhat = self.model(x_test)
                predictions.append(yhat.detach().numpy())
                values.append(y_test.detach().numpy())

        return predictions, values

    def plot_losses(self):
        plt.plot(self.train_losses, label="Training loss")
        plt.plot(self.val_losses, label="Validation loss")
        plt.legend()
        plt.title("Losses")
        plt.show()
        plt.close()



# ------------------------------------------------------------------------------------------------------
# Helper functions defined below
# ------------------------------------------------------------------------------------------------------


# functions (1 of 2) to get data stored in the drive into the dummy csv file
def new_csv(df, df_append_list=[]):
  new_df = df[['x_gt', 'y_gt', 'x_gps', 'y_gps', 'IMU_x', 'IMU_y', 'gyro']]
  new_df.to_csv('../datasets/rnn_dummy.csv', index=False)
  
  if len(df_append_list) != 0:
    for _df in df_append_list:
        append_df = _df[['x_gt', 'y_gt', 'x_gps', 'y_gps', 'IMU_x', 'IMU_y', 'gyro']]
        append_df.to_csv('../datasets/rnn_dummy.csv', mode='a', index=False, header=False)

# functions (2 of 2) to get data stored in the drive into the dummy csv file
def prepareDataset(N, training, files_to_append):
  if training:
    folder = 'Training_Data/Training_Set_'
    folderType = 'Training Data #'
  else:
    folder = 'Validation_Data/Validation_Set_'
    folderType = 'Validation Data #'

  path = '../datasets/' + folder + str(N) + '.csv'
  df = pd.read_csv(path)

  dfs_to_append = []
  if len(files_to_append) != 0:
    for file_path in files_to_append:
      dfs_to_append.append(pd.read_csv(file_path))
  
  new_csv(df, dfs_to_append)



# Gets the dataset in the current rnn dummy file, and returns a tuple
# containing the scaled dataframe, the original, and the scalars in the order
# x1, x2, x3, x4, x5, y1, y2
def readData():
    # read data from .csv to dataframe
    df_rnn = pd.read_csv('../datasets/rnn_dummy.csv')
    #df_rnn.drop(df_rnn.columns[[0]], axis=1, inplace=True)                         
    # Create a copy of the raw .csv file the RNN model will use (don't want to overwrite original values)
    rnn_df_scaled = df_rnn.copy()
    # Scalers for all the features (independent of each other)
    x1_scaler = preprocessing.MinMaxScaler() # GPS X coordinate scalar
    x2_scaler = preprocessing.MinMaxScaler() # GPS Y coordinate scalar
    x3_scaler = preprocessing.MinMaxScaler() # Accelerometer X scalar
    x4_scaler = preprocessing.MinMaxScaler() # Accelerometer Y scalar
    x5_scaler = preprocessing.MinMaxScaler() # Gyro angular velocity reading scalar
    # Scalers for the outputs x,y (independent of each other)
    y1_scaler = preprocessing.MinMaxScaler() # x_gt scaler
    y2_scaler = preprocessing.MinMaxScaler() # y_gt scaler
    # Normalize X and Y of the RNN data to have values between 0 and 1
    rnn_df_scaled['x_gt'] = y1_scaler.fit_transform(df_rnn['x_gt'].to_numpy().reshape(-1, 1))
    rnn_df_scaled['y_gt'] = y2_scaler.fit_transform(df_rnn['y_gt'].to_numpy().reshape(-1, 1))
    # X (features)
    rnn_df_scaled['x_gps'] = x1_scaler.fit_transform(df_rnn['x_gps'].to_numpy().reshape(-1, 1))
    rnn_df_scaled['y_gps'] = x2_scaler.fit_transform(df_rnn['y_gps'].to_numpy().reshape(-1, 1))
    rnn_df_scaled['IMU_x'] = x3_scaler.fit_transform(df_rnn['IMU_x'].to_numpy().reshape(-1, 1))
    rnn_df_scaled['IMU_y'] = x4_scaler.fit_transform(df_rnn['IMU_y'].to_numpy().reshape(-1, 1))
    rnn_df_scaled['gyro'] = x5_scaler.fit_transform(df_rnn['gyro'].to_numpy().reshape(-1, 1))

    return (rnn_df_scaled, df_rnn, x1_scaler, x2_scaler, x3_scaler, x4_scaler, x5_scaler, y1_scaler, y2_scaler)





# Takes in the scaled and unscaled dataset, getting the features and labels of 
# each instance in a tuple, and returning a tuple of tuples
def splitData(rnn_df_scaled, df_rnn, train_size=1624, val_size=541, test_size=541):
    train_data_scaled = rnn_df_scaled[:train_size]
    val_data_scaled = rnn_df_scaled[train_size : train_size + val_size]
    test_data_scaled = rnn_df_scaled[train_size + val_size : ]

    X_train_scaled = train_data_scaled.filter(['x_gps','y_gps','IMU_x','IMU_y','gyro'], axis=1)
    Y_train_scaled = train_data_scaled.filter(['x_gt','y_gt'], axis=1)
    train_tuple_scaled = (X_train_scaled, Y_train_scaled)
    X_test_scaled = test_data_scaled.filter(['x_gps','y_gps','IMU_x','IMU_y','gyro'], axis=1)
    Y_test_scaled = test_data_scaled.filter(['x_gt','y_gt'], axis=1)
    test_tuple_scaled = (X_test_scaled, Y_test_scaled)
    X_val_scaled = val_data_scaled.filter(['x_gps','y_gps','IMU_x','IMU_y','gyro'], axis=1)
    Y_val_scaled = val_data_scaled.filter(['x_gt','y_gt'], axis=1)
    val_tuple_scaled = (X_val_scaled, Y_val_scaled)

    train_data = df_rnn[:train_size]
    val_data = df_rnn[train_size : train_size + val_size]
    test_data = df_rnn[train_size + val_size : ]

    X_train_raw = train_data.filter(['x_gps','y_gps','IMU_x','IMU_y','gyro'], axis=1)
    Y_train_raw = train_data.filter(['x_gt','y_gt'], axis=1)
    train_tuple = (X_train_raw, Y_train_raw)
    X_test_raw = test_data.filter(['x_gps','y_gps','IMU_x','IMU_y','gyro'], axis=1)
    Y_test_raw = test_data.filter(['x_gt','y_gt'], axis=1)
    test_tuple = (X_test_raw, Y_test_raw)
    X_val_raw = val_data.filter(['x_gps','y_gps','IMU_x','IMU_y','gyro'], axis=1)
    Y_val_raw = val_data.filter(['x_gt','y_gt'], axis=1)
    val_tuple = (X_val_raw, Y_val_raw)

    return (train_tuple_scaled, test_tuple_scaled, val_tuple_scaled, train_tuple, test_tuple, val_tuple)






# Takes in a set of 3 tuples containing the scaled inputs and splits the data 
# into randomly shuffled batches, returned  in a tuple of train, validation, 
# test, and unshuffled test
def dataLoading(training_data, validation_data, testing_data, batch_size=24):
    train_features = torch.Tensor(training_data[0].to_numpy())
    train_targets = torch.Tensor(training_data[1].to_numpy())

    test_features = torch.Tensor(testing_data[0].to_numpy())
    test_targets = torch.Tensor(testing_data[1].to_numpy())

    val_features = torch.Tensor(validation_data[0].to_numpy())
    val_targets = torch.Tensor(validation_data[1].to_numpy())

    train = TensorDataset(train_features, train_targets)
    val = TensorDataset(val_features, val_targets)
    test = TensorDataset(test_features, test_targets)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader_one = DataLoader(test, batch_size=1, shuffle=False, drop_last=True)
    return (train_loader, val_loader, test_loader, test_loader_one)




# Functions to transform the predictions back to original scale
def inverse_transform(df, y1_scaler, y2_scaler):
    df_gt = df.filter(['x_gt','y_gt'], axis=1)
    df_pred = df.filter(['x_pred','y_pred'], axis=1)

    arr_x_gt = y1_scaler.inverse_transform(df_gt['x_gt'].to_numpy().reshape(-1 ,1))
    arr_y_gt = y2_scaler.inverse_transform(df_gt['y_gt'].to_numpy().reshape(-1 ,1)) 

    arr_x_pred = y1_scaler.inverse_transform(df_pred['x_pred'].to_numpy().reshape(-1 ,1))
    arr_y_pred = y2_scaler.inverse_transform(df_pred['y_pred'].to_numpy().reshape(-1 ,1)) 

    df_gt['x_gt'] = arr_x_gt
    df_gt['y_gt'] = arr_y_gt

    df_pred['x_pred'] = arr_x_pred
    df_pred['y_pred'] = arr_y_pred

    return pd.concat([df_gt, df_pred], axis=1)


# Formats the prediction results
def format_predictions(predictions, values, df_test, y1_scaler, y2_scaler):
    vals = np.concatenate(values, axis=0)
    preds = np.concatenate(predictions, axis=0)
    df_result = pd.DataFrame(data={'x_gt': vals[:,0], 'y_gt': vals[:,1], 'x_pred': preds[:,0], 'y_pred': preds[:,1]}, index=df_test.head(len(vals)).index)
    df_result = df_result.sort_index()
    df_result = inverse_transform(df_result, y1_scaler, y2_scaler)
    return df_result


# Function that calculates metrics
def calculate_metrics(df):
    return {
            'mae_x' : mean_absolute_error(df.x_gt, df.x_pred),
            'rmse_x' : mean_squared_error(df.x_gt, df.x_pred) ** 0.5,
            'r2_x' : r2_score(df.x_gt, df.x_pred),
            'mae_y' : mean_absolute_error(df.y_gt, df.y_pred),
            'rmse_y' : mean_squared_error(df.y_gt, df.y_pred) ** 0.5,
            'r2_y' : r2_score(df.y_gt, df.y_pred)
            }
def gps_metrics(df):
    return {
            'mae_x' : mean_absolute_error(df.x_gt, df.x_gps),
            'rmse_x' : mean_squared_error(df.x_gt, df.x_gps) ** 0.5,
            'r2_x' : r2_score(df.x_gt, df.x_gps),
            'mae_y' : mean_absolute_error(df.y_gt, df.y_gps),
            'rmse_y' : mean_squared_error(df.y_gt, df.y_gps) ** 0.5,
            'r2_y' : r2_score(df.y_gt, df.y_gps)
            }



def plotResults(df_rnn, df_result):

    plt.figure(figsize=(12, 8))
    for row in df_result.iterrows():
        plt.scatter(row[1][0], row[1][1], color = 'r', s = 12)
        plt.scatter(row[1][2], row[1][3], color = 'b', s = 12)

    plt.show()



def trainModelStartToFinish(trial, lr, epochs, num_layers, hidden_size, append_dataset_list=[], N=0, display=True):
    batch_size = 1024
    n_epochs = epochs
    #n_epochs = 100
    # learning_rate = 1e-4
    learning_rate = lr
    #weight_decay = weight_decay
    n_features=5

    prepareDataset(N, True, append_dataset_list)
    processedData = readData()

    rnn_df_scaled = processedData[0]
    df_rnn = processedData[1]
    y1_scaler = processedData[7]
    y2_scaler = processedData[8]
    #big_tuple = splitData(rnn_df_scaled, df_rnn)

    #manually change the sizes
    #big_tuple = splitData(rnn_df_scaled, df_rnn, train_size=13530, val_size=5412, test_size=5412)
    big_tuple = splitData(rnn_df_scaled, df_rnn, train_size=1624, val_size=541, test_size=541)
    training = big_tuple[0]
    testing = big_tuple[1]
    validation = big_tuple[2]
    X_test_scaled = testing[0]

    train_loader, val_loader, test_loader, test_loader_one = dataLoading(training, validation, testing)
    model = RNN_Model(input_size=n_features, output_size=2, hidden_dim=hidden_size, n_layers=num_layers)
    loss_fn = nn.MSELoss(reduction="mean")
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    opt = Trainer(model=model, loss_fn=loss_fn, optimizer=optimizer)
    opt.train(trial, train_loader, val_loader, batch_size=batch_size, n_epochs=n_epochs, n_features=n_features)

    predictions, values = opt.evaluate(test_loader_one, batch_size=1, n_features=n_features)
    df_result = format_predictions(predictions, values, X_test_scaled,  y1_scaler, y2_scaler)
    result_metrics = calculate_metrics(df_result)
    result_metrics_gps = gps_metrics(df_rnn)
    opt.plot_losses()
    plotResults(df_rnn, df_result)

    mae_x = result_metrics['mae_x']
    mae_y = result_metrics['mae_y']
    rmse_x = result_metrics['rmse_x']
    rmse_y = result_metrics['rmse_y']
    r2_x = result_metrics['r2_x']
    r2_y = result_metrics['r2_y']

    mae_x_gps = result_metrics_gps['mae_x']
    mae_y_gps = result_metrics_gps['mae_y']
    rmse_x_gps = result_metrics_gps['rmse_x']
    rmse_y_gps = result_metrics_gps['rmse_y']
    r2_x_gps = result_metrics_gps['r2_x']
    r2_y_gps = result_metrics_gps['r2_y']

    print(f"MAE X: {result_metrics['mae_x']}\t MAE Y: {result_metrics['mae_y']}")

    return (model, result_metrics)



def run_model(trial=0,lr=0.00008, epochs=100, num_layers=1, hidden_size=256, append_dataset_list=[], N=0, display=True):
    return trainModelStartToFinish(trial=trial, lr=lr, epochs=epochs, num_layers=num_layers,
        hidden_size=hidden_size, append_dataset_list=append_dataset_list, N=N, display=display)



def main():
    args = sys.argv[0:]
    argc = 1
    run_model(trial=0,lr=0.00008, epochs=100, num_layers=1, hidden_size=256, append_dataset_list=[], N=0, display=True)

if __name__ == '__main__':
    main()
