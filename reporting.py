'''
reporting.py
Rafael Guerra
January 2022

This script creates a confusion matrix

'''

# Import libraries
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os


#Genereate confusion matrix
def generate_cmatrix(config_path,filename):
    
    #Load config.json and get path variables
    with open(config_path,'r') as f:
        config = json.load(f) 

    # Import Paths
    output_folder_path = config['output_folder_path']
    test_data_path = config['test_data_path'] 
    output_model_path = config['output_model_path'] 
    prod_deployment_path = config['prod_deployment_path'] 
    
    # Load dataset
    test_data = pd.read_csv(os.getcwd()+test_data_path+'testdata.csv')
    
    #calculate a confusion matrix using the test data and the deployed model
    X = test_data[['lastmonth_activity','lastyear_activity','number_of_employees']].values.reshape(-1,3)
    y = test_data['exited'].values.reshape(-1,1).ravel()
    
    # Load model
    with open(os.getcwd()+output_model_path+'/'+'trainedmodel.pkl', 'rb') as file:
        model = pickle.load(file)
    
    #write the confusion matrix to the workspace
    plot_confusion_matrix(model, X, y)
    
    #write the confusion matrix to the workspace
    plt.savefig(os.getcwd()+output_model_path+ filename)

if __name__ == '__main__':
    generate_cmatrix('config.json','confusion_matrix.png')