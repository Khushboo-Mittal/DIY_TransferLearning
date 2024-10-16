# META DATA - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

     # Developer details: 
        # Name: Harshita and Prachi
        # Role: Architects
    # Version:
        # Version: V 1.0 (20 September 2024)
            # Developers: Harshita and Prachi
            # Unit test: Pass
            # Integration test: Pass
     
    # Description: This code snippet preprocesses the data for a machine learning model by scaling
    # numerical columns, encoding categorical columns, and extracting date components for before feeding it to train the model
        # PostgreSQL: Yes

# CODE - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    # Dependency: 
        # Environment:     
            # Python 3.11.5
            # Pandas 2.2.2
            # Scikit-learn 1.5.0
import numpy as np
import pandas as pd     # Importing pandas for data manipulation
from sqlalchemy import create_engine
from sklearn.preprocessing import LabelEncoder  # Importing tools for data preprocessing
import spacy
from pymongo import MongoClient

def preprocess_mongo_data(data):
    data.drop_duplicates(inplace=True)
    data.dropna(inplace=True)
    # Load spaCy model
    nlp = spacy.load("en_core_web_sm") 

    def preprocess(text):
        doc = nlp(text)
        filtered_tokens = []
        for token in doc:
            # Skip stopwords and punctuation
            if not token.is_stop and not token.is_punct:
                filtered_tokens.append(token.lemma_)  # Lemmatize the token
                
        return " ".join(filtered_tokens)
    
    data['Preprocessed Text'] = data['tweet_content'].apply(preprocess) 
    # Encode categorical columns
    le = LabelEncoder() # Initialize the LabelEncoder
    data['sentiment'] = le.fit_transform(data['sentiment']) # Encode sentiment column
    return data


def load_and_preprocess_data (mongo_host, mongo_port, mongo_db):

    # Connect to MongoDB
    client = MongoClient(f'mongodb://{mongo_host}:{mongo_port}')
    db = client[mongo_db]
    collection = db['Twitter_training']

    # Load MongoDB data into a DataFrame
    data_mongodb = pd.DataFrame(list(collection.find()))

    # If needed, you can drop the MongoDB's default "_id" column
    data_mongodb.drop('_id', axis=1, inplace=True)  # Uncomment if you want to remove the "_id" column
    
    # Preprocess data
    data_mongo_processed = preprocess_mongo_data(data_mongodb) # Preprocess PostgreSQL data
    return data_mongo_processed

