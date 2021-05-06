import sys

# import statements
import pandas as pd
from sqlalchemy import create_engine

# define the data loading function
def load_data(messages_filepath, categories_filepath):
    # read 'disaster_messages.csv' in to messages
    messages = pd.read_csv(messages_filepath)
    # read 'disaster_categories.csv' in to categories
    categories = pd.read_csv(categories_filepath)
    # inner join these tow dataframes on 'id' column
    df = messages.merge(categories, left_on = ['id'], right_on = ['id'], how = 'inner')
    # return the obtained dataframe
    return df

# define cleaning data function
def clean_data(df):
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';',expand = True)
    # select the first row of the categories dataframe
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = pd.Series(categories.values[0,:]).apply(lambda x : x[:-2])
    # rename the columns of `categories`    
    categories.columns = category_colnames
    
    #convert category values to numerical values (0 and 1)
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1:]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    # drop the original categories column from `df`
    df = df.drop(columns = ['categories'])
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis = 1)
    # drop duplicates
    df = df.drop_duplicates()
    
    return df

#define save data function
def save_data(df, database_filepath):
    #create an engine connect to sqlite
    engine = create_engine(str('sqlite:///' + database_filepath))
    #save as database
    df.to_sql(database_filepath[5:-3], engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()