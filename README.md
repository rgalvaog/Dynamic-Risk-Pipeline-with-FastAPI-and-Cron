## Dynamic Risk Assessment System

This project is a great example of how to link up individual components of a Machine Learning pipeline in such a way that it can be checked by colleagues through API access and run automatically with crontab in case there is any new data to be ingested or any model drift detected.

### Platforms / Technologies
Technologies used in the project
* [GitHub](github.com)
* [crontab](https://man7.org/linux/man-pages/man5/crontab.5.html)
* [Python](https://https://www.python.org)

### Getting Started

#### Set up miniconda environment

This project was setup using miniconda. To setup the enviornment run:
```conda env create -f conda_local.env```

#### Full pipeline run

To run the entire pipeline, run:
```fullprocess.py```

You can also debug individual parts of the pipeline such as `ingestion.py` to verify the data ingestion process, or `training.py` to look at the logistic regression parameters set for this project.


##### The pipeline (steps)

###### Step 1: Data Ingestion 
- `ingestion.py`: Script checks for new datasets and if there are any, they are added into the `'finaldata.csv'` file which will then be run against the diagnostics script to see if further training is needed.

###### Step 2: F1 Score Comparison 
- `diagnostics.py`: We examine the F1 score of the new predictions with the new data. If the F1 score is lower than the previous F1 score, then we conclude model drift has occured and therefore, we will retrain the model. 

###### Step 3: Retraining
- `training.py`: Retrain model with logistic regression.

###### Step 4: Diagnostics and Reporting

- In this step we run `diagnostics.py` and `reporting.py` to examine runtime, old dependencies, summary statistics, and a confusion matrix for the model.
	
##### Step 5: API 
At any point, one can also run `app.py` in one shell, and run `python apicall.py` on another one to run the pipline. Additionally, one may simply run `app.py` in one shell and call up `curl 127.0.0.1:8000/endpoint` where endpoint refers to one of the endpoints on the app.py file.