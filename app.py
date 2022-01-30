'''

app.py
Rafael Guerra
January 2022

This script modularizes the pipeline into an API with endpoints.

'''

# Import libraries
from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import json
import os
import subprocess
from diagnostics import model_predictions, dataframe_summary, calculate_missing_data_proportions, execution_time, outdated_packages_list
from scoring import score_model


#Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('production_config.json','r') as f:
    config = json.load(f) 

model_path = os.path.join(config['output_model_path'])     
dataset_csv_path = os.path.join(config['output_folder_path']) 
prediction_model = pickle.load(open(os.path.join(os.getcwd()+ model_path+ "trainedmodel.pkl"), "rb"))
test_data_path = os.path.join(config['test_data_path']) 
test_dataset = os.getcwd()+test_data_path+'testdata.csv'


def readpandas(filename):
    dataset=pd.read_csv(filename)
    return dataset

#######################Prediction Endpoint
@app.route("/prediction",methods=['GET','OPTIONS'])
def predict():
    data_filename = request.args.get("inputdata")
    predictions = model_predictions('config.json')
    return str(predictions)

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def score():    
    f1 = score_model('config.json')
    print(f1)
    return str(f1)

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def stats():
    return str(dataframe_summary('config.json'))

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnose():        
    timing_stat = execution_time()
    missing_data = calculate_missing_data_proportions('config.json')
    outdated_report = outdated_packages_list()
    return str((timing_stat,missing_data))

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
