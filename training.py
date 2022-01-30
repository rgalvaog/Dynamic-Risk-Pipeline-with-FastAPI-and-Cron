'''
training.py
Rafael Guerra
January 2022

This script conducts the model training.

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


#################Function for training the model
def train_model(config_file):
       
    #Load config.json and get path variables
    with open(config_file,'r') as f:
        config = json.load(f) 
    
    output_folder_path = config['output_folder_path'] 
    output_model_path = config['output_model_path']
    final_sales = pd.read_csv(os.getcwd()+output_folder_path+'finaldata.csv')
    
    #Please note I have removed the multi_class warning
    #It was throwing an error to the creation of the .pkl file
    logit = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                    intercept_scaling=1, l1_ratio=None, max_iter=100,n_jobs=None, penalty='l2',
                    random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                    warm_start=False)
    
    #fit the logistic regression to your data
    X = final_sales.loc[:,['lastmonth_activity','lastyear_activity','number_of_employees']].values.reshape(-1, 3)
    y = final_sales.loc[:,['exited']].values.reshape(-1, 1).ravel()
    model = logit.fit(X, y)
    
    #write the trained model to your workspace in a file called trainedmodel.pkl
    filehandler = open(os.getcwd()+output_model_path+'trainedmodel.pkl', 'wb')
    pickle.dump(model, filehandler)

if __name__ == '__main__':
    train_model('config.json')
