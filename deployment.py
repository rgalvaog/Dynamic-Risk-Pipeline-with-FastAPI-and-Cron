'''

deployment.py
Rafael Guerra
January 2022

This script conducts the deployment of the model into the production path.

'''

# Import libraries
from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json

# Deployment function
def store_model_into_pickle(config_path):
      
    #Load config.json and correct path variable
    with open(config_path,'r') as f:
        config = json.load(f) 

    output_folder_path = config['output_folder_path']
    prod_deployment_path = config['prod_deployment_path'] 
    model_path = config['output_model_path']
    
    #Copy Pickle
    with open(os.getcwd()+model_path+'trainedmodel.pkl', 'rb') as file:
        model = pickle.load(file)

    # Copy model score(latestscore.txt)
    with open(os.getcwd()+model_path+'latestscore.txt', 'rb') as file:
        latest_score = file.read()

    # Copy record of ingested data (ingestedfiles.txt)
    with open(os.getcwd()+output_folder_path+'ingestedfiles.txt', 'rb') as file:
        ingested_files = file.read()

    # Copy Model to Production Folder
    filehandler = open(os.getcwd()+prod_deployment_path+'trainedmodel.pkl', 'wb')
    pickle.dump(model, filehandler)

    # Copy Latest Scores into Production Folder
    with open(os.getcwd()+prod_deployment_path+'latestscore.txt', 'wb') as f:
        f.write(latest_score)

    # Copy Ingested Files into Production Folder
    with open(os.getcwd()+prod_deployment_path+'ingestedfiles.txt', 'wb') as f:
        f.write(ingested_files)    

if __name__ == '__main__':
    store_model_into_pickle('config.json')