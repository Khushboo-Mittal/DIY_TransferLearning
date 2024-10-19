# META DATA - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

     # Developer details: 
        # Name: Harshita Jangde
        # Role: Architects
    # Version:
        # Version: V 1.0 (19 October 2024)
            # Developers: Harshita Jangde
            # Unit test: Pass
            # Integration test: Pass
     
    # Description: This code snippet contains functions to split preprocessed data into test, validation,
    # and super validation and store it in a MongoDB database.
        # MongoDB: Yes

# CODE - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    # Dependency: 
        # Environment:     
            # Python 3.11.5


# Importing necessary libraries
import pandas as pd                                      # For data manipulation
from sklearn.model_selection import train_test_split     # To split data into train, test, validation, and super validation sets
from pymongo import MongoClient                          # For using MongoDB as a cache to store the split data
import pickle       # For serializing and deserializing data for storage in MongoDB
from db_utils import load_data_from_mongodb
import streamlit as st

def connect_to_mongodb(host, port, db_name):
    # Connect to MongoDB
    client = MongoClient(host=host, port=port)
    db = client[db_name]
    return db

def split_data(data):
    """
    Split the data into training, testing, validation, and super validation sets.
    """
    # Perform stratified sampling on the entire dataframe before splitting X and y
    data = data.groupby('sentiment', group_keys=False).apply(lambda x: x.sample(frac=5000/75682, random_state=42))
    X = data['Preprocessed Text']  # Use the entire preprocessed data as features
    y = data['sentiment']
    # Split the data into train, test, validation, and super validation sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)

    # Split the temporary set into validation (50%) and test (50%) to get 10% each
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.4,stratify=y_temp, random_state=42)
    return X_train, X_test,X_val,y_train,y_val,y_test

def store_to_mongo(data, db, collection_name):
    """
    Store the data into a MongoDB collection.
    """
    collection = db[collection_name]  # Select the collection
    collection.insert_one({'data': data})  # Insert the data into the collection

def save_split_data(db, X_train, X_test, X_val, y_train,y_val,y_test):
    """
    Store the split data (train, test, val, superval) into MongoDB.
    """
    store_to_mongo(pickle.dumps(X_train), db, 'x_train')
    store_to_mongo(pickle.dumps(X_test), db, 'x_test')
    store_to_mongo(pickle.dumps(X_val), db, 'x_val')
    store_to_mongo(pickle.dumps(y_train), db, 'y_train')
    store_to_mongo(pickle.dumps(y_val), db, 'y_val')
    store_to_mongo(pickle.dumps(y_test), db, 'y_test')


def split_data_and_store(mongodb_host, mongodb_port, mongodb_db, data):
    """
    Main function to preprocess, split, and store the data into MongoDB.
    """
    client = MongoClient(host=mongodb_host, port=mongodb_port)
    db = client[mongodb_db]
    collection = db["preprocessed_tweet_data"]
    # Fetch data from MongoDB
    data = list(collection.find())
    data = pd.DataFrame(data)
    st.write(f"Data loaded successfully with columns: {list(data.columns)}")
    
    # Split data into train, test, validation, and super validation sets
    X_train, X_test,X_val,y_train,y_val,y_test = split_data(data)
    
    # Save split data into MongoDB
    save_split_data(db, X_train, X_test,X_val,y_train,y_val,y_test)
    
    print('Data preprocessed, and split successfully!')

