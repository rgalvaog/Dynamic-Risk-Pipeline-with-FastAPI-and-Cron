'''
fullprocess.py
Rafael Guerra
January 2022

This script contains the full model pipeline, checking for new data, model drift, and re-running the model if needed.

'''

# Import libraries
import pandas as pd
import json
import os
import sys
import pickle
import ingestion
import training
import scoring
import deployment
import diagnostics
import reporting
import apicall

# Load production config.json
with open('production_config.json', 'r') as f:
    config = json.load(f)

# Relevant paths
ingested_files_path = os.getcwd() + config['prod_deployment_path'] + 'ingestedfiles.txt'
latest_f1_score_path = os.getcwd() + config['prod_deployment_path'] + 'latestscore.txt'
new_data_path = os.getcwd() + config['input_folder_path']
prod_deployment_path = config['prod_deployment_path']
test_data_path = config['test_data_path']


#Checking for whether there is new data
def check_for_new_data():
    
    new_ingested_data = []

    with open(ingested_files_path, 'r') as f:
        ingested_data = f.readlines()
    
    # Convert string list to list
    ingested_data = ingested_data[0].strip("[]")
    ingested_data = ingested_data.replace(" ","")
    ingested_data = ingested_data.replace("'","")
    ingested_data = ingested_data.split(',')
    
    # Create list with new data
    data_to_be_ingested = []
    for entry in os.listdir(new_data_path):
        data_to_be_ingested.append(new_data_path + '/' + entry)
    
    newData = False
    if ingested_data!=data_to_be_ingested:
        newData = True
    else:
        newData = False
    
    return newData

# Ingest data
def ingest_data():
    new_data = ingestion.merge_multiple_dataframes('production_config.json')
    

# Check for model drift
def check_model_drift():
    with open(latest_f1_score_path, 'r') as f:
        latest_f1_score = f.readlines()
    
    # Retrieve latest F1 score
    latest_f1_score = latest_f1_score[0]
    latest_f1_score = float(latest_f1_score)
    
    # Make predictions with model
    new_scoring_data = pd.read_csv(os.getcwd()+config['output_folder_path']+'finaldata.csv')
    scoring.score_model('production_config.json')
    
    # Check score again
    with open(os.getcwd() + config['output_model_path']+ "latestscore.txt", 'r') as f:
        new_f1_score = f.readlines()
    
    # Retrieve new F1 score
    new_f1_score = new_f1_score[0]
    new_f1_score = float(new_f1_score)

    # Check for model drift
    modelDrift = False
    if new_f1_score < latest_f1_score:
        modelDrift = True
    else:
        modelDrift = False
    
    return modelDrift


# Rerun training, scoring, and deployment
def redeploy():
    
    # Retrain model with new training data
    new_data = pd.read_csv(os.getcwd()+config['output_folder_path']+'finaldata.csv')
    model = training.train_model('production_config.json')
    
    # Rescore model
    testdata = pd.read_csv(os.getcwd()+test_data_path+'testdata.csv')
    new_model_score = scoring.score_model('production_config.json')
    
    # Store model in production path
    deployment.store_model_into_pickle('production_config.json')

# Re-run diagnosing
def rediagnose():
    
    # From diagnostics
    diagnostics.dataframe_summary(os.getcwd()+config['output_folder_path']+'finaldata.csv')
    diagnostics.calculate_missing_data_proportions(os.getcwd()+config['output_folder_path']+'finaldata.csv')
    diagnostics.execution_time()
    diagnostics.outdated_packages_list()
    
    # Generate Confusion Matrix
    reporting.generate_cmatrix('production_config.json','confusionmatrix2.png')
    
    # Call API
    apicall("apireturns2.txt")    

if __name__ == '__main__':
    
    if check_for_new_data() is True:
        ingest_data()
    else:
        sys.exit()
        
    check_model_drift()
   
    if check_model_drift() is True:
        redeploy()
        rediagnose()
    else:
        sys.exit()