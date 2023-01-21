# AI Multi-Sensor Fusion
 Repository for the team's coding on the project for Aerospace

Pre-requisites:
        - pip install the following
        - pip install tsaug
        - pip install pytorch-lightning
        - pip install wandb
        - pip install pyproj

How to run:

    Run the following srcipts from within the 'AI-Multi_Sensor-Fusion/scripts' folder

    1. Data visualization
        To view the GroundTruthAGL plot you can simply run the following from the 
        scripts folder:
            python data_visualization.py
        To specify the path you can run the follwing:
            python data_visualization.py -path ../datasets/GroundTruthAGL.csv
        
        To view the onBoardGPS plot you need to specify both the path and the type which
        by default are the following:
            python data_visualization.py -path ../datasets/onBoardGPS.csv -type onBoard


        Note: These function assumes that the labels for the ground truth and gps
        positions are as follows  ' x_gt',' y_gt',' x_gps',' y_gps', ' lon', ' lat'. 
        There is an extra space in at the start of the labels

    2. Data Preprocessing
        The data_preprocessing.py script is used to generate the LR_processed_data.csv file.
        The script matches rows from the onBoardGPS.csv file to rows from the GroundTruthAGL.csv
        file by their imgid value into one localized LR_processed_data.csv file. The script can be run as follows:
		python data_preprocessing.py

        Note: Coordinates are translated from lat, lon to x, y (in the onBoardGPS.csv file)
        using the pyproj library

    3. Data Augmentation
        This script runs the data augmentation process. the file
        'LR_processed_data.csv' should be in the same file as this, as that file
        contains the preprocessed data drawn from the dataset.
        With no arguements, or a single numerical argument, the script generates a 
        new dataset, plots out the old and new datasets, and asks the user if they 
        want to save it after the plots are closed. The numerical argument is a scaling
        factor for the noise of the GPS data. Ex:
            python Data_Augmentation.py 1.2
        If the argument "view" is passed, followed by a path, the script will display 
        information about the dataset. Ex:
            python Data_Augmentation.py view c:/datatsets/Training_Data/Training_Set_14.csv
        Otherwise, if a string argument that is not view is given, it will create a folder
        with that name and fill it with multiple augmented datasets. If a numerical 
        argument is given after that folder, it will be used as a scale for the 
        intensity of noise used to generate the GPS data, as above. Ex:
            python Data_Augmentation.py datasets
            python Data_Augmentation.py smallError 0.3
            python Data_Augmentation.py noisy 1.5


    4. RNN
        The rnn script runs the RNN model. The rnn model was developed in the beginning 
	of the creation process but is mainly still here for original comparison reasons.The LSTM has proven to produce 
	better results so that model is the final product. Running this model will still be helpful for improvement 
	possibilities and visual comparisons. 
            python rnn.py 

    5. LSTM
        The lstm script runs our lstm model and logs the results to WandB. The script will prompt
        the user to log into wandb or create an account if not already signed in. The -n flag sets
        the number of trials the user would like optuna to optimize for. In WandB the script 
        generates two projects, MSF which contains numerical information about the run and 
        MSF Optuna which contains visuals about the runs
            python lstm.py -n 100

    6. Outlier Detection (low thresholded KNN)
	This script plots a comparison of prediction data and true data when given a path. The current model is currently has the 
	outlier quartile range set to 0.75, which means that it will have at most 25% of the data be outliers. Another condition was
	added that checks if the distance is a minimum distance to be an error. Both of these are adjustable variables based on what 
	the user is attempting to observe. By default it uses the Ground Truth AGL file from the dataset, but if you want to evaluate
	points of concentrated error in models, give it a path to a model prediction and the parameter -m after said path. Ex:
		python outlier.py c:/datatsets/GroundTruthAGL.csv
