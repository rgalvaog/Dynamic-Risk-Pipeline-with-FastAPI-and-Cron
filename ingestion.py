'''
ingestion.py
Rafael Guerra
January 2022

This script conducts the data ingestion process.

'''

# Import Libraries
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

# Merge Multiple Dataframes
def merge_multiple_dataframes(config_file):
    
    #Load config.json and get input and output paths
    with open(config_file,'r') as f:
        config = json.load(f) 

    input_folder_path = config['input_folder_path']
    output_folder_path = config['output_folder_path']
    
    # Instantiate empty dataframe
    dataframe = pd.DataFrame(columns=['corporation','lastmonth_activity','lastyear_activity','number_of_employees','exited'])

    # Compile files that are in directory together
    directory = os.getcwd()+input_folder_path
    filenames = []
    for file in os.listdir(directory):
        currentdf = pd.read_csv(directory+file)
        filenames.append(directory+file)
        dataframe = dataframe.append(currentdf).reset_index(drop=True)

    # Drop Duplicate Rows
    final_dataframe=dataframe.drop_duplicates()    

    # Write final file to specified output folder
    final_dataframe.to_csv(os.getcwd()+output_folder_path+'finaldata.csv',index=False)
        
    # Create log file
    with open(os.getcwd()+output_folder_path+'ingestedfiles.txt', 'w') as f:
        f.write(str(filenames))


if __name__ == '__main__':
    merge_multiple_dataframes('config.json')