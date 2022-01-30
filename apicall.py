'''

apicalls.py
Rafael Guerra
January 2022

This script calls the API endpoints

'''

# Import libraries
import subprocess
import json
import os

def apicall(filename):
    
    #Specify a URL that resolves to your workspace
    URL = "http://127.0.0.1/"

    with open('production_config.json','r') as f:
        config = json.load(f) 

    model_path = config['output_model_path']    

    #Call each API endpoint and store the responses
    response1=subprocess.run(['curl', '127.0.0.1:8000/prediction?inputdata=testdata.csv'],capture_output=True).stdout
    response2=subprocess.run(['curl', '127.0.0.1:8000/scoring'],capture_output=True).stdout
    response3=subprocess.run(['curl', '127.0.0.1:8000/summarystats'],capture_output=True).stdout
    response4=subprocess.run(['curl', '127.0.0.1:8000/diagnostics'],capture_output=True).stdout
    response_list = [response1,response2,response3,response4]
    response_list_str = str(response_list)

    with open(os.getcwd()+model_path+filename, 'w') as f:
        f.write(response_list_str)    

if __name__ == '__main__':
    apicall('apireturns.txt')
