# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. If you are runing this script locally, Go to http://localhost/0.0.0.0:3001/

### Introduction

This project contains a machine learning pipeline developed to categorize real-life messages people sent out during emergeny situations so that these messages can be forwarded to an appropriate relief agency. This project also includes a web app that an emergeny worker can input a new message and get the classification results in several categories.

### Data understanding

There are two datasets: 
	1. 'disaster_messages.csv' contains 4 columns with the first column as the message id, the 2nd to 4th columns as the message content, the message origin and the message genre 
	2. 'disaster_categories' which contains all the message ids and their categories. Notice that the categories column (1 sigle column) contains all the categories for each message, and if a message belongs to a specific category the value follow that category is 1, othersiwe the value is 0.
	
Some initial thought on this data is to combine these 2 datasets together with the message id as the common key and then assign message content, genre and origin as X with category as Y and then build model base on this.

### Prepare Data

#### Data Cleaning

Data cleaning process in this project mainly involves create 36 individual categorical columns from the one single category column. Since there is no empty values in these two dataset, we don't need to worry about frop or fill null value in this particular project. We still need to drop dupicalte data if there is any in the dataset. The details of data cleaning process is included in the python script 'process_data.py'

#### NLP

Before we can perform any 

###