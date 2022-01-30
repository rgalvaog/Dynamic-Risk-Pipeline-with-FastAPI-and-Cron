'''

diagnostics.py
Rafael Guerra
January 2022

This script conducts diagnostics in the model such as reading into oudated packages and calculating how long it takes to run functions.

'''

# Import libraries
import pandas as pd
import numpy as np
import timeit
import os
import json
import pickle
import subprocess


#Function to get model predictions
def model_predictions(config_path):
     
    #Load config.json and get folder paths
    with open(config_path,'r') as f:
        config = json.load(f) 

    test_data_path = config['test_data_path']
    model_path = config['output_model_path']

    #Load model
    with open(os.getcwd()+model_path+'/'+'trainedmodel.pkl', 'rb') as file:
        model = pickle.load(file)
    
    # Load test dataset
    test_dataset = pd.read_csv(os.getcwd()+test_data_path+'testdata.csv')
    X = test_dataset.loc[:,['lastmonth_activity','lastyear_activity','number_of_employees']].values.reshape(-1, 3)
    y = test_dataset.loc[:,['exited']].values.reshape(-1, 1).ravel()

    # Make predictions
    y_predictions= list(model.predict(X))

    # Return Predictions
    return y_predictions

##################Function to get summary statistics
def dataframe_summary(config_path):
    
    #Load config.json and get folder paths
    with open(config_path,'r') as f:
        config = json.load(f) 
        
    output_folder_path = config['output_folder_path']
        
    # Initiate blank summary statistics list
    summary_statistics=[]

    # Get numerical columns from final Dataset (from ingesteddata)
    final_dataset = pd.read_csv(os.getcwd()+output_folder_path+'finaldata.csv')
    num_cols = ['lastmonth_activity','lastyear_activity','number_of_employees']
    for column in num_cols: 
        summary_statistics.append(np.mean(final_dataset[column]))
        summary_statistics.append(np.median(final_dataset[column]))
        summary_statistics.append(np.std(final_dataset[column]))
    
    # Return Summary Statistics Table
    return summary_statistics

##################Function to calculate percentage of missing values in each column
def calculate_missing_data_proportions(config_path):

    #Load config.json and get folder paths
    with open(config_path,'r') as f:
        config = json.load(f) 
        
    output_folder_path = config['output_folder_path']
    
    # Load in dataset
    final_dataset = pd.read_csv(os.getcwd()+output_folder_path+'/'+'finaldata.csv')
    
    # Calculate percentage of NA values by column
    nas=list(final_dataset.isna().sum())
    napercents=[nas[i]/len(final_dataset.index) for i in range(len(nas))]

    # Return NA Percents
    return napercents

##################Function to get timings
def execution_time():

    #calculate timing of training.py
    training_starttime = timeit.default_timer()
    os.system('python3 ingestion.py')
    training_timing=timeit.default_timer() - training_starttime
    
    #  and ingestion.py
    ingestion_starttime = timeit.default_timer()
    os.system('python3 ingestion.py')
    ingestion_timing=timeit.default_timer() - ingestion_starttime

    # Create timing list
    timing_list = [training_timing,ingestion_timing]

    # Return timing list
    return timing_list

##################Function to check dependencies
def outdated_packages_list():
    
    # Use PIP to check the current and latest versions of our packages
    outdated_packages = subprocess.check_output(['pip', 'list', '--outdated'])
    
    # Write table with output from pip command to outdated.packages.txt
    with open('outdated_packages.txt', 'wb') as f:
        f.write(outdated_packages)


if __name__ == '__main__':
    model_predictions('config.json')
    dataframe_summary('config.json')
    calculate_missing_data_proportions('config.json')
    execution_time()
    outdated_packages_list()





    

