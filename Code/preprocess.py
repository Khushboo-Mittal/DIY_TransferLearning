# META DATA - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

     # Developer details: 
        # Name: Harshita 
        # Role: Architects
    # Version:
        # Version: V 1.0 (19 October 2024)
            # Developers: Harshita 
            # Unit test: Pass
            # Integration test: Pass
     
    # Description: This code snippet preprocesses the data for a machine learning model before feeding it to train the model
        # MongoDB: Yes

# CODE - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    # Dependency: 
        # Environment:     
            # Python 3.11.5
            # Pandas 2.2.2

import pandas as pd     # Importing pandas for data manipulation
from sklearn.preprocessing import LabelEncoder  # Importing tools for data preprocessing
import spacy
from pymongo import MongoClient
import streamlit as st

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
    le = LabelEncoder() # Initialize the LabelEncoder
    data['sentiment'] = le.fit_transform(data['sentiment']) # Encode sentiment column
    return data

    
def load_and_preprocess_data(mongodb_host, mongodb_port, mongodb_db):
    client = MongoClient(host=mongodb_host, port=mongodb_port)
    db = client[mongodb_db]
    collection = db["tweet_data"]
    new_collection = db["preprocessed_tweet_data"]

    # Fetch data from MongoDB and convert it to a pandas DataFrame
    data = list(collection.find())
    df = pd.DataFrame(data)

    # If the collection exists and has data, skip preprocessing
    if new_collection.estimated_document_count() > 0:
        st.write(f"preprocessed_tweet_data already exists with data. Skipping preprocessing.")
        return
    data_preprocessed = preprocess_mongo_data(df)
    
    data_dict = data_preprocessed.to_dict(orient="records")
    
    # Insert the data into the MongoDB collection
    new_collection.insert_many(data_dict)