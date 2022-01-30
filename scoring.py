'''
scoring.py
Rafael Guerra
January 2022

This script conducts the scoring for the model.

'''

# Import packages
from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json


#Scoring Model function
def score_model(config_file):
    
    #Load config.json and get path variables
    with open(config_file,'r') as f:
        config = json.load(f) 

    model_path = config['output_model_path']
    testdata = pd.read_csv(os.getcwd()+config['test_data_path']+'testdata.csv')
    
    # Load independent variables in X, dependent variable in Y
    X = testdata[['lastmonth_activity','lastyear_activity','number_of_employees']].values.reshape(-1,3)
    y = testdata['exited'].values.reshape(-1,1).ravel()
    
    # Load model
    with open(os.getcwd()+model_path+'trainedmodel.pkl', 'rb') as file:
        model = pickle.load(file)
    
    # Predict
    predicted=model.predict(X)
    f1score=metrics.f1_score(predicted,y)
    
    with open(os.path.join(os.getcwd()+model_path+ "latestscore.txt"), "w") as f:
        f.write(str(f1score))
    
    return f1score
    
if __name__ == '__main__':
    score_model('config.json')