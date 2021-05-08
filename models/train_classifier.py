import sys

#import libraries

import sqlalchemy as db

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report


# create load data function
def load_data(database_filepath):
    
    #creat engine and connnection to the sql database
    # 'database_filepath' point to the input datafile name and path  
    engine = db.create_engine(str('sqlite:///'+database_filepath))
    connection = engine.connect()
    
    #load sql database to panda dataframe

    df = pd.read_sql(str("SELECT * FROM "+database_filepath[5:-3]), con=connection)
    # set 'message' column as X, and set all of the categorical columns as y
    X = df['message']
    y = df[df.columns[4:]]
    # get column names
    category_names = list(df.columns[4:])
    
    return X, y, category_names
    

# define the function to tokenized the text
def tokenize(text):
    #tokenized sentences
    tokens = word_tokenize(text)
    #create lammatizer
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        #lower case, lammatize and remove 
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        #append to clean_tok list
        clean_tokens.append(clean_tok)
    #return the cleaned tokens
    return clean_tokens

# defined the funtion to build to model
def build_model():
    
    #build pipeline
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    # here we use Random Forest Classifier
    # more classifier can be explore such as ...
    ('clf', RandomForestClassifier())
    ])
    
    #set parameters for GridSearch # only selected 1 here to reduce program processing time
    #parameters = {
        #'vect__max_features': (None,5000),
        #'tfidf__smooth_idf': (True, False),
    #}
    
    #cv = GridSearchCV(pipeline, param_grid=parameters)

    return pipeline

# define the function to evaluate the model
def evaluate_model(model, X_test, Y_test, category_names):
    # get the predicted value from the test set
    Y_pred = model.predict(X_test)
    
    n = 0
    for col in category_names:
        # print category name
        print(col)
        # display the classfication report for this category
        print(classification_report(Y_test[col], Y_pred[:,n]))
        n+=1

    #print("\nBest Parameters:", model.best_params_)

# save this model as a pickle file for the webapp
def save_model(model, model_filepath):
    import pickle
    # 'model_filepath' points to the desired path and filename of the pickle file
    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))


def main():
    
    # the function takes in 3 argument including this python file
    if len(sys.argv) == 3:
        #database_filepath, model_filepath are the last 2 arguments
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
