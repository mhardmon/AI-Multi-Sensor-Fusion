import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    args = sys.argv[0:]

    if len(args) == 1:
        plot_original_GTdataset('../datasets/GroundTruthAGL.csv')
    else:
        argc = 1
        path ='../datasets/GroundTruthAGL.csv'
        file = 'GT'

        while argc < len(args):
            if args[argc] == '-path':
                assert(argc + 1 <= len(args) - 1)
                path = args[argc + 1]
            elif args[argc] == '-type':
                assert(argc + 1 <= len(args) - 1)
                file = 'onBoard'
            argc += 1

        assert(file == 'GT' or file == 'onBoard')

        if file == 'GT':
            plot_original_GTdataset(path)
        else:
            plot_original_onBoarddataset(path)
        

def plot_original_GTdataset(path):

    # read data from .csv in to dataframe
    df = pd.read_csv(path)

    # plot lines
    plt.plot(df[' x_gt'], df[' y_gt'], label = 'Ground Truth Position', color='red')
    plt.plot(df[' x_gps'], df[' y_gps'], label = 'GPS Position', color='blue')

    plt.title('GroundTruthAGL Plot')
    plt.legend()
    plt.show()

def plot_original_onBoarddataset(path):

    # read data from .csv in to dataframe
    df = pd.read_csv(path)

    # plot line
    plt.plot(df[' lon'], df[' lat'], label = 'Coordinates', color='red')

    plt.title('onBoardGPS Plot')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()