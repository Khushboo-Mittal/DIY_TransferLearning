# META DATA - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

     # Developer details: 
        # Name: Harshita and Tanisha
        # Role: Architects
    # Version:
        # Version: V 1.0 (20 September 2024)
            # Developers: Harshita and prachi
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
from db_utils import load_FaceMask_data_from_mongodb as load_data_from_mongodb

def connect_to_mongodb(host, port, db_name):
    # Connect to MongoDB
    client = MongoClient(host=host, port=port)
    db = client[db_name]
    return db

def split_data(data):
    """
    Split the data into training, testing, validation, and super validation sets.
    """
    # Convert the MongoDB data to a DataFrame if it's not already
    X = data['Preprocessed Text']  # Use the entire preprocessed data as features
    y = data['sentiment']
    # Split the data into train, test, validation, and super validation sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)

    # Split the temporary set into validation (50%) and test (50%) to get 10% each
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    return X_train, X_test,X_val,y_train,y_val,y_test

def store_to_mongo(data, db, collection_name):
    """
    Store the data into a MongoDB collection.
    """
    collection = db[collection_name]  # Select the collection
    collection.insert_one({'data': data})  # Insert the data into the collection

def save_split_data(db, X_train, X_test, X_val, X_superval):
    """
    Store the split data (train, test, val, superval) into MongoDB.
    """
    store_to_mongo(pickle.dumps(X_train), db, 'x_train')
    store_to_mongo(pickle.dumps(X_test), db, 'x_test')
    store_to_mongo(pickle.dumps(X_val), db, 'x_val')
    store_to_mongo(pickle.dumps(X_superval), db, 'x_superval')


def split_data_and_store(mongodb_host, mongodb_port, mongodb_db, data):
    """
    Main function to preprocess, split, and store the data into MongoDB.
    """
    # Connect to MongoDB
    db = connect_to_mongodb(mongodb_host, mongodb_port, mongodb_db)
    
    # Split data into train, test, validation, and super validation sets
    X_train, X_test,X_val,y_train,y_val,y_test = split_data(data)
    
    # Save split data into MongoDB
    save_split_data(db, X_train, X_test,X_val,y_train,y_val,y_test)
    
    print('Data preprocessed, and split successfully!')

